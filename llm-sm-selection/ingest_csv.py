import pandas as pd
import json
from pathlib import Path
import fire

def convert_csv_to_json(csv_path, num_rows, output_name):
    """
    Converte um arquivo CSV para JSON, limitando o número de linhas.
    Salva o resultado na pasta llm-sm-selection/data/.

    Args:
        csv_path (str): Caminho para o arquivo CSV de entrada.
        num_rows (int): Quantidade de linhas a serem processadas.
        output_name (str): Nome do arquivo final (ex: articles.json).
    """
    # Define o diretório de saída relativo a este script
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / output_name

    try:
        # Carrega o CSV limitando as linhas
        print(f"📖 Lendo arquivo: {csv_path}...")
        df = pd.read_csv(csv_path, nrows=int(num_rows))

        # Trata valores nulos para compatibilidade com JSON (NaN -> None/null)
        df = df.replace("", None)
        df = df.astype(object).where(pd.notnull(df), None)

        # Converte o DataFrame para uma lista de dicionários
        articles = df.to_dict(orient='records')

        # Salva em formato JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        print(f"✅ Sucesso! {len(articles)} linhas processadas.")
        print(f"📁 Arquivo salvo em: {output_path}")

    except Exception as e:
        print(f"❌ Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    fire.Fire(convert_csv_to_json)
