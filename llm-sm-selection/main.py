import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src.config import ConfigManager
from src.llms.gemini_ternario import GeminiLLMV2
from src.llms.chatgpt_ternario import ChatGPTLLM
from src.llms.consensus_ternario import ConsensusLLM


def main():
    # 1. Carregamento de ambiente e configurações
    load_dotenv()
    
    manager = ConfigManager()
    llm_params = manager.get_llm_configs()
    metadata = manager.get_experiment_metadata()
    dataset = llm_params.get("dataset")
    provider = llm_params.get("provider", "google")

    # 2. Configuração de Chave de API e Classe por provedor
    if provider == "openai":
        classifier = ChatGPTLLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=llm_params["model"],
            config=llm_params
        )
    elif provider == "google":
        classifier = GeminiLLMV2(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name=llm_params["model"],
            config=llm_params
        )
    elif provider == "consensus":
        classifier = ConsensusLLM(
            openai_key=os.getenv("OPENAI_API_KEY"),
            gemini_key=os.getenv("GOOGLE_API_KEY"),
            gpt_model=llm_params.get("gpt_model"),
            gemini_model=llm_params.get("gemini_model"),
            config=llm_params
        )
    else:
        raise ValueError(f"Provider {provider} not supported.")

    data_path = Path(f"data/{dataset}")
    with open(data_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # 3. Inicialização do Nome do Modelo para Salvar
    model_name = llm_params.get("model", "consensus")

    # 4. Execução do Experimento
    articles = articles[:100]
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
