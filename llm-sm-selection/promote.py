import shutil
import json
from pathlib import Path
from datetime import datetime
import fire  # ← importa o Fire

def promote_to_result(exp_folder_name: str, version_label: str):
    source = Path(f"{exp_folder_name}")
    target = Path(f"results/{version_label}")

    if not source.exists():
        print(f"❌ Experimento não encontrado: {source}")
        return

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "result.json", target / "result.json")
    shutil.copy(source / "config_used.yaml", target / "config_used.yaml")

    manifest = {
        "promoted_at": datetime.now().isoformat(),
        "label": version_label
    }
    with open(target / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Experimento promovido com sucesso para: {target}")

if __name__ == "__main__":
    fire.Fire(promote_to_result)