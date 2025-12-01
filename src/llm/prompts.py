"""Prompt template loader."""

from pathlib import Path
from typing import Optional


_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """
    Load a prompt template from prompts/ directory.
    
    Args:
        name: Prompt file name (without .txt extension)
        
    Returns:
        Prompt template content as string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = _PROMPTS_DIR / f"{name}.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    return prompt_path.read_text(encoding="utf-8")

