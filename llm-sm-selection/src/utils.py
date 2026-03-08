# src/utils.py
import json
from pathlib import Path
from typing import Any, Dict, List

def read_json_file(file_path: Path) -> Dict[str, Any]:
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def write_json_file(data: Dict[str, Any], file_path: Path):
    """Writes a dictionary to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error writing to {file_path}: {e}")

def get_json_files(directory: Path) -> List[Path]:
    """Gets all JSON files in a directory."""
    return list(directory.glob("*.json"))
