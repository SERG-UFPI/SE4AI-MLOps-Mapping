# Systematic Mapping with LLMs

This project provides a simple and extensible pipeline to automate the initial screening phase of a systematic mapping study using multiple Large Language Models (LLMs). It reads a collection of academic articles in JSON format and applies pre-defined inclusion and exclusion criteria to classify each one.

## Features

- **Multi-LLM Screening**: Screen articles using different providers (OpenAI, Anthropic, Google Gemini) simultaneously.
- **Configurable Criteria**: Easily define and change inclusion/exclusion criteria in a `config.yaml` file without touching the code.
- **Extensible Architecture**: A simple `BaseLLM` interface makes it straightforward to add support for new LLM providers.
- **Command-Line Interface**: Run the entire pipeline from your terminal.

## Project Structure

```
systematic-mapping/
├── data/
│   ├── input/          # Place your raw JSON articles here
│   └── output/         # Screened results will be saved here
├── notebooks/          # Jupyter notebooks for analysis
├── src/
│   ├── llms/           # Wrappers for different LLM providers
│   ├── pipeline.py     # Main pipeline orchestration logic
│   ├── criteria.py     # Logic for loading screening criteria
│   └── utils.py        # Helper functions for file I/O
├── config.yaml         # Configuration for API keys, models, and criteria
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How to Use

### 1. Installation

First, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd systematic-mapping
pip install -r requirements.txt
```

### 2. Configuration

Before running the pipeline, you need to configure your API keys and screening criteria in `config.yaml`:

1.  **API Keys**: Add your API keys for the LLM providers you want to use under the `api_keys` section.
2.  **LLM Models**: Configure which models to use in the `llm_configs` list. Make sure the `provider` name matches a key in `api_keys`.
3.  **Screening Criteria**: Define your study's `inclusion` and `exclusion` criteria under the `screening_criteria` section.

### 3. Input Data

Place your articles in the `data/input/` directory. Each article should be a separate JSON file. The pipeline expects each file to contain at least an `id`, `title`, and `abstract`.

Example `article_1.json`:
```json
{
  "id": "smith2023learning",
  "title": "A Novel Approach to Learning from Unstructured Data",
  "abstract": "This paper presents a new machine learning technique..."
}
```

### 4. Running the Pipeline

Execute the pipeline from the root directory of the project:

```bash
python -m src.pipeline --input data/input/ --output data/output/
```

The script will:
- Read each JSON file from the input directory.
- Send the title and abstract to each configured LLM for screening.
- Save one JSON result file per article per LLM in the output directory. The result includes the decision (`include`/`exclude`), a reason, and the model used.

## How to Add a New LLM

To add support for a new LLM provider (e.g., "MyNewLLM"):

1.  **Create a new file**: Add a file in the `src/llms/` directory, e.g., `src/llms/mynewllm.py`.
2.  **Implement the interface**: Inside the new file, create a class that inherits from `BaseLLM` and implements the `screen_article` method.

    ```python
    # src/llms/mynewllm.py
    from .base import BaseLLM
    
    class MyNewLLM(BaseLLM):
        def screen_article(self, article, criteria_prompt):
            # Your logic to call the new LLM's API
            prompt = self._create_prompt(article, criteria_prompt)
            
            # ... call API ...
            
            # Return the result in the standard format
            return {
                "id": article.get("id"),
                "decision": "include", # or "exclude"
                "reason": "...",
                "llm_used": self.model,
            }
    ```

3.  **Register the new class**: In `src/llms/__init__.py`, import your new class and add it to the `get_llm_provider` factory function.

    ```python
    # src/llms/__init__.py
    # ... other imports
    from .mynewllm import MyNewLLM # Import your new class

    def get_llm_provider(provider_name):
        provider_map = {
            "openai": OpenAILLM,
            "anthropic": AnthropicLLM,
            "google": GoogleLLM,
            "mynewllm": MyNewLLM, # Add your new provider here
        }
        # ...
    ```

4.  **Update `config.yaml`**: Add the new provider's API key and model configuration to `config.yaml`.

## Analysis Notebook

The `notebooks/analysis.ipynb` notebook provides an example of how to load the screening results, compare the decisions made by different LLMs, and calculate agreement scores (e.g., Cohen's Kappa). This can help you understand the consistency and reliability of the automated screening process.
