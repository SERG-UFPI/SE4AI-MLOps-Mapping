# src/pipeline.py
import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, List

from src.llms import get_llm_provider, BaseLLM
from src.criteria import load_criteria, get_criteria_prompt
from src.utils import get_json_files, read_json_file, write_json_file

def load_llms_from_config(config_path: Path) -> List[BaseLLM]:
    """Loads LLM providers based on the config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    api_keys = config.get("api_keys", {})
    llm_configs = config.get("llm_configs", [])
    
    llms = []
    for llm_config in llm_configs:
        provider_name = llm_config["provider"]
        model = llm_config["model"]
        api_key = api_keys.get(provider_name)
        
        if not api_key:
            raise ValueError(f"API key for {provider_name} not found in config.yaml")
        
        LLMProvider = get_llm_provider(provider_name)
        llms.append(LLMProvider(model=model, api_key=api_key))
        
    return llms

def run_pipeline(input_dir: Path, output_dir: Path, config_path: Path):
    """
    Orchestrates the screening pipeline.
    
    1. Loads criteria and LLMs from config.
    2. Reads articles from the input directory.
    3. For each article, screens it with each LLM.
    4. Saves the results to the output directory.
    """
    print("Starting pipeline...")

    # Load criteria and create prompt
    criteria = load_criteria(config_path)
    if not criteria.get("inclusion") and not criteria.get("exclusion"):
        print("Warning: No screening criteria found. The results may not be meaningful.")
    criteria_prompt = get_criteria_prompt(criteria)

    # Load LLMs
    try:
        llms = load_llms_from_config(config_path)
        if not llms:
            print("No LLMs configured in config.yaml. Exiting.")
            return
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading LLMs: {e}")
        return

    # Get article files
    article_files = get_json_files(input_dir)
    if not article_files:
        print(f"No JSON articles found in {input_dir}.")
        return

    print(f"Found {len(article_files)} articles to screen with {len(llms)} LLM(s).")

    # Process each article
    for article_path in article_files:
        article = read_json_file(article_path)
        if not article or "id" not in article:
            print(f"Skipping invalid article: {article_path.name}")
            continue

        print(f"Screening article: {article.get('id', 'N/A')}")
        
        # Screen with each LLM
        for llm in llms:
            result = llm.screen_article(article, criteria_prompt)
            
            # Save the result
            output_filename = f"{article['id']}_{llm.model.replace('/', '_')}.json"
            output_path = output_dir / output_filename
            write_json_file(result, output_path)

    print("Pipeline finished successfully.")

def main():
    """Main function to run the pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Systematic Mapping Pipeline using LLMs.")
    parser.add_argument(
        "--input",
        type=Path,
        default="data/input",
        help="Directory containing input JSON articles."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/output",
        help="Directory to save the screening results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.config)

if __name__ == "__main__":
    main()
