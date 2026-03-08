# src/criteria.py
from typing import Dict, List
import yaml
from pathlib import Path

def load_criteria(config_path: Path = Path("config.yaml")) -> Dict[str, List[str]]:
    """
    Loads inclusion and exclusion criteria from a YAML config file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        A dictionary with 'inclusion' and 'exclusion' criteria.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get("screening_criteria", {})
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading criteria from {config_path}: {e}")
        return {"inclusion": [], "exclusion": []}

def get_criteria_prompt(criteria: Dict[str, List[str]]) -> str:
    """
    Generates a formatted string of criteria for the LLM prompt.
    """
    prompt = "Screening Criteria:\n"
    prompt += "Inclusion Criteria:\n"
    for criterion in criteria.get("inclusion", []):
        prompt += f"- {criterion}\n"
    
    prompt += "\nExclusion Criteria:\n"
    for criterion in criteria.get("exclusion", []):
        prompt += f"- {criterion}\n"
        
    return prompt
