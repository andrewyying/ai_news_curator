"""OpenAI client wrapper for LLM and embedding calls."""

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import settings


# Initialize OpenAI client
_client = None


def get_client() -> OpenAI:
    """Get or create OpenAI client instance."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
    return _client


def call_llm_json(
    prompt: str | List[Dict[str, str]],
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Call OpenAI ChatCompletion API and return JSON response.
    
    Args:
        prompt: User prompt string or list of message dicts
        system: System message
        model: Model name (defaults to settings.openai_model)
        temperature: Sampling temperature
        
    Returns:
        Parsed JSON response as dict
    """
    client = get_client()
    model_name = model or settings.openai_model
    
    # Prepare messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        messages.extend(prompt)
    
    # Make API call
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    
    # Parse JSON response
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON response: {content}") from e


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    client = get_client()
    
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    
    return [item.embedding for item in response.data]

