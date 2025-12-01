# MPCS 57200 Generative AI - Final Project: AI News Curator

This is an AI-powered news curation tool that automatically fetches, classifies, scores, and summarizes tech/AI news from multiple new sources, generating structured daily reports.

## Features

- **Multi-source News Aggregation**: Fetches news from Hacker News, TechCrunch, OpenAI Blog, NVIDIA Blog, arXiv, and more of users' choices
- **Intelligent Classification**: Uses LLM to categorize news into 7 categories (AI Models, AI Infrastructure, AI Research, AI Policy, Developer Tools, Tech Business, Other)
- **Impact Scoring**: Assigns importance scores (1-5) to each news item with AI-generated explanations
- **Smart Deduplication**: Uses embedding-based clustering to identify and merge similar news stories
- **Automated Summarization**: Generates concise summaries for each news cluster
- **Evaluation Framework**: Includes experimental evaluation comparing zero-shot vs few-shot classification accuracy
- **Structured Reports**: Outputs well-formatted Markdown daily reports

## Installation

### Prerequisites

- Python 3.10+
- Available OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/andrewyying/ai_news_curator.git
   cd ai_news_curator
   ```

2. Install and activate virtual environment
    ```bash
    python -m venv venv
    # If Mac / linux
    source venv/bin/activate
    # If Windows
    venv\Scripts\activate
    ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Configure environment variables:
   
   Create a `.env` file in the project root:
   ```bash
   # Copy the example file
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Run Daily News Curation Pipeline

Generate a daily news report for today:

```bash
python -m cli run-daily
# or
ai-news-curator run-daily
```

Or specify a date:
```bash
python -m cli run-daily --date 2024-11-30
# or
ai-news-curator run-daily --date 2024-11-30
```

The generated report will be saved in the `/reports` directory as `reports/YYYY-MM-DD.md`.


### Advanced Configuration

You can customize the LLM and embedding models by editing the `.env` file:
```env
OPENAI_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

Default news sources are configured in `src/config.py`. To use a custom list of RSS feeds, override the `RSS_FEEDS` environment variable with comma-separated URLs (please make sure the source supports RSS format):
```env
RSS_FEEDS=https://hnrss.org/frontpage,https://techcrunch.com/feed/,https://example.com/feed
```

## Course Concepts and Technologies Applied in the Project

This project demonstrates the application of several key concepts and technologies we discussed in the course:

### 1. Pre-trained Models

The foundation of this project rests on leveraging pre-trained models (by default, gpt-4o-mini) for multiple downstream tasks including text classification, impact scoring, and summarization. The project also employs pre-trained embedding models (text-embedding-3-small) for semantic similarity computation during the deduplication phase.

### 2. Prompt Engineering

Prompt engineering is central to this project's design. The project implements zero-shot classification, where carefully crafted prompts enable the model to categorize news articles into predefined categories without any task-specific training examples. Noticeable improvements in classification accuracy were observed during prompt iterations, demonstrating how the knowledge embedded in pre-trained models can be accessed and directed through strategic prompt design. 

The project also investigates whether zero-shot or few-shot learning is optimal for this application. See the [Zero-shot vs Few-shot](#zero-shot-vs-few-shot) section below for evaluation results.

### 3. Embeddings and Semantic Search

The project utilizes text embeddings to enable semantic understanding and similarity computation, which is essential for the deduplication and clustering functionality. When processing news articles, each article's title and content are converted into dense vector representations using OpenAI's embedding model. These embeddings capture semantic meaning in a high-dimensional space, where articles discussing similar topics or events are positioned closer together regardless of their exact wording.

The project employs cosine similarity to measure the semantic distance between these vector representations, enabling the identification of duplicate or highly similar news stories that might have been reported by different sources or with different headlines. The clustering algorithm groups articles with similarity scores above a threshold, effectively merging redundant information while preserving the diversity of perspectives.

### 4. Responsible AI

Responsible AI considerations are woven throughout the project's design, reflecting the ethical obligations that come with deploying AI systems in real-world applications. The project includes "Responsible AI Notes" in the summary generation process, where the model is explicitly instructed to flag potential concerns such as hype, safety issues, misinformation, bias, or privacy implications.

The prompts used throughout the pipeline explicitly instruct the model to be conservative in its assessments, to avoid speculation, and to refrain from making claims about financial returns or stock prices. The emphasis on factual accuracy and explicit uncertainty acknowledgment demonstrates responsible AI practices, helping prevent the model from generating confident but potentially incorrect information. This is particularly important given that the core value of this application is helping users summarize information and save time.



## Zero-shot vs Few-shot

### Evaluation Methodology

To determine whether zero-shot or few-shot learning is more effective for news classification, we conducted a systematic evaluation using manually labeled ground truth data. The evaluation compares both approaches on the same dataset to provide a fair comparison.

**Evaluation Dataset**: The evaluation uses manually labeled samples from `src/evaluation/sample_labels.json`, which contains news items with ground truth category labels across all 7 categories.

**Evaluation Metrics**:
- Overall accuracy: Percentage of correctly classified items
- Per-category accuracy: Performance breakdown by category
- Improvement analysis: Quantitative comparison between approaches

### Running the Evaluation

Compare zero-shot vs few-shot classification accuracy:
```bash
python -m evaluation.eval_classification
# or 
python src/evaluation/eval_classification.py
```

This will:
- Load manually labeled samples from `src/evaluation/sample_labels.json`
- Run both zero-shot and few-shot classification on the same dataset
- Calculate accuracy metrics and per-category performance
- Generate a detailed evaluation report in `src/evaluation/classification_results.md`

### Experimental Results

Our evaluation on 15 manually labeled news samples yielded the following results:

**Overall Performance**:
- **Zero-shot**: 93.33% accuracy (14/15 correct)
- **Few-shot**: 100.00% accuracy (15/15 correct)
- **Improvement**: +6.67%

**Per-Category Analysis**:

| Category | Zero-shot | Few-shot |
|----------|-----------|----------|
| AI Infrastructure & Hardware | 100.00% (3/3) | 100.00% (3/3) |
| AI Models | 100.00% (3/3) | 100.00% (3/3) |
| AI Policy & Regulation | 100.00% (2/2) | 100.00% (2/2) |
| AI Research | 100.00% (2/2) | 100.00% (2/2) |
| Developer Tools & Platforms | 100.00% (2/2) | 100.00% (2/2) |
| Tech Business & Strategy | 100.00% (2/2) | 100.00% (2/2) |
| Other | 0.00% (0/1) | 100.00% (1/1) |

### Key Findings

1. **Zero-shot achieves high accuracy**: With 93.33% overall accuracy, zero-shot learning demonstrates excellent performance across most categories. All categories except "Other" achieved 100% accuracy with zero-shot.

2. **Limited improvement from few-shot**: While few-shot learning achieved perfect accuracy, the improvement is marginal (+6.67%) and comes at a significant cost increase. The only misclassification was in the "Other" category, where a non-AI tech news item (iPhone release) was incorrectly classified as "Tech Business & Strategy" by zero-shot.

3. **Cost-effectiveness analysis**: 
   - Zero-shot uses significantly fewer tokens (no examples in prompt)
   - Few-shot requires additional tokens for examples (~200-300 tokens per classification)
   - For a production system processing hundreds of news items daily, the cost difference is substantial
   - The 6.67% accuracy gain may not justify the increased cost, especially given that zero-shot already achieves 93.33% accuracy

4. **Category-specific insights**: Zero-shot performs exceptionally well on well-defined categories (AI Models, AI Research, etc.), suggesting that the model's pre-trained knowledge is sufficient for most classification tasks in this domain.

### Conclusion

Based on our experimental results:

- **Production recommendation**: Zero-shot learning is recommended for this application. The 93.33% accuracy is highly satisfactory for news classification, and the cost savings are significant when processing large volumes of news daily.

- **Edge case handling**: The single misclassification in the "Other" category suggests that zero-shot may struggle with ambiguous cases that don't clearly fit predefined categories. For such cases, a hybrid approach could be considered:
  - Use zero-shot as the default method
  - Apply few-shot or manual review only for low-confidence predictions or items classified as "Other"

- **Cost-benefit trade-off**: The marginal 6.67% improvement from few-shot does not justify the substantial increase in API costs for most use cases. The cost per classification with few-shot is approximately 2-3x higher than zero-shot, making zero-shot the more practical choice for scalable news curation systems.


## Project Structure

```
ai_news_curator/
├── src/                            # Source code
│   ├── cli.py                      # Command-line interface
│   ├── main.py                     # Main pipeline orchestrator
│   ├── config.py                   # Configuration management
│   ├── models.py                   # Pydantic data models
│   ├── cache.py                    # Caching utilities
│   ├── fetchers/                   # News fetching modules
│   │   └── rss_fetcher.py          # RSS feed parser
│   ├── llm/                        # LLM client wrapper
│   │   ├── client.py               # OpenAI API wrapper
│   │   └── prompts.py              # Prompt template loader
│   ├── pipeline/                   # Processing pipeline
│   │   ├── classify.py             # News classification
│   │   ├── impact.py               # Impact scoring
│   │   ├── deduplicate.py          # Clustering & deduplication
│   │   ├── summarize.py            # Summary generation
│   │   └── report.py               # Markdown report generation
│   └── evaluation/                 # Evaluation modules
│       ├── eval_classification.py  # Classification evaluation
│       ├── sample_labels.json      # Labeled samples
│       └── label_schema.md         # Labeling guidelines
├── prompts/                        # Prompt templates
│   ├── classifier_prompt_zero_shot.txt
│   ├── classifier_prompt_few_shot.txt
│   ├── impact_prompt.txt
│   └── summary_prompt.txt
├── data/                           # Data storage
│   ├── raw_news/                   # Raw fetched news (JSON)
│   ├── curated/                    # Processed news (JSON)
│   └── cache/                      # LLM response cache
├── reports/                        # Generated daily reports (Markdown)
├── pyproject.toml                  # Project configuration
└── README.md                       # This file
```
