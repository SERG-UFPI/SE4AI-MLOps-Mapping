import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src.config import ConfigManager
from src.llms.gemini import GeminiLLM


def main():
    # 1. Carregamento de ambiente e configurações
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    manager = ConfigManager()
    llm_params = manager.get_llm_configs()
    metadata = manager.get_experiment_metadata()

    data_path = Path("data/artigos.json")
    with open(data_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # 3. Inicialização do Classificador
    model_name = llm_params["model"]
    classifier = GeminiLLM(api_key=api_key, model_name=model_name, config=llm_params)

    # 4. Execução do Experimento
    articles = articles[:2]
    print(f"Iniciando experimento: {metadata['experiment_id']}")
    results = classifier.batch_classify(articles, manager.get_inlusion_criteria())

    # 5. Criação da estrutura de pastas para o resultado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider = llm_params.get("provider", "google")

    # Caminho: experiments/google/20260308_2100-gemini-3.1-flash/
    output_dir = Path(
        f"experiments/{provider}/{timestamp}-{model_name.replace('/', '-')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Salvando os resultados e o snapshot do config.yaml utilizados
    result_file = output_dir / "result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Copia o config.yaml para dentro da pasta do experimento para auditoria
    shutil.copy("config.yaml", output_dir / "config_used.yaml")

    print("\n✅ Experimento concluído!")
    print(f"📁 Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
