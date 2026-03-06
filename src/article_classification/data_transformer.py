import pandas as pd


class DataTransformer:
    """
    Classe responsável por transformar dados entre formatos
    e aplicar tratamentos básicos.
    """

    def __init__(self, encoding="utf-8"):
        self.encoding = encoding

    def load_csv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, encoding=self.encoding)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica limpezas básicas nos dados.
        """
        # remover espaços extras em colunas string
        for col in df.select_dtypes(include=["object", "string"]):
            df[col] = df[col].str.strip()

        return df

    def dataframe_to_json(self, df: pd.DataFrame, orient="records") -> str:
        """
        Converte DataFrame para string JSON.
        """
        return df.to_json(orient=orient, force_ascii=False)

    def csv_to_json(self, input_csv: str, output_json: str = None):
        """
        Pipeline completo: CSV -> limpeza -> JSON
        """
        df = self.load_csv(input_csv)
        df = self.clean_data(df)

        if output_json:
            df.to_json(output_json, orient="records", force_ascii=False, indent=4)
            return output_json

        return self.dataframe_to_json(df)


if __name__ == "__main__":
    transformer = DataTransformer()
    transformer.csv_to_json("data/Artigos IEE Xplore.csv", "./data/articles.json")