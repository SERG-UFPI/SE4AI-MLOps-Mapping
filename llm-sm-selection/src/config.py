import yaml
from pathlib import Path


class ConfigManager:
    def __init__(self, config_name="config.yaml"):
        self.config_path = Path(__file__).parent.parent / config_name
        self.config = self._load_yaml()

    def _load_yaml(self):
        """Carrega o arquivo YAML com tratamento de erro."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado em: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_eligibility_criteria(self):
        """Retorna os critérios de inclusão e exclusão mapeados."""
        return self.config.get("eligibility_criteria", {})

    def get_inlusion_criteria(self):
        """Retorna os critérios de inclusão mapeados."""
        return self.get_eligibility_criteria().get("inclusion", [])

    def get_experiment_metadata(self):
        """Retorna os metadados do experimento atual."""
        return self.config.get("metadata", {})

    def get_llm_configs(self):
        """Retorna as configurações específicas do LLM."""
        return self.config.get("llm_configs", {})
