
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
import time
from datetime import datetime


class SLRClassifier():

    def __init__(self, api_key:str, model_name:str="gemini-1.5-pro"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
    

    def _build_system_instruction(self, item:dict) -> str:
        return (
            "Assume you are a software engineering researcher. "
            "Conducting a systematic literature review (SLR). Consider the title, "
            "abstract and keywords of a primary study\n"
            "Title: " + item["Título"] + "\n"
            "Abstract: " + item["Abstract"] + "\n"
        )

    def _build_prompt(self, criteria: str) -> str:
        return (
            f"Using a 1-7 Likert scale (1 - Strongly disagree, 2 - Disagree, 3 - Somewhat disagree, 4 - Neither agree nor disagree, 5 - Somewhat agree, 6 - Agree, and 7 - Strongly agree) rate your agreement with the question (only number): {criteria}"
        )
    
    def evaluate(self, item: dict, criteria: str) -> tuple[str, int]:
        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1,
            top_p=0.1,
            system_instruction=self._build_system_instruction(item),
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=self._build_prompt(criteria),
            config=config
        )
        time.sleep(10) 
        score = response.text.strip()[0]
        tokens = response.usage_metadata.total_token_count
        return score, tokens
    
    def run(self, data: list[dict], criteria: dict) -> list[dict]:
        criteria_names = list(criteria.keys())
        n_criteria = len(criteria_names)

        token_cols = [f"Tokens {name}" for name in criteria_names]
        columns = ["Title"] + criteria_names + ["Execution time (s)"] + token_cols + ["Total Tokens"]
        result_np = np.empty((len(data), len(columns)), dtype=object)

        print(f"Starting... Using model: {self.model_name}")
    
        for i, item in enumerate(data):
            start_time = time.perf_counter()
            result_np[i, 0] = item["Título"]

            scores = []
            tokens_list = []

            for j, (name, description) in enumerate(criteria.items()):
                score, tokens = self.evaluate(item, description)
                scores.append(int(score))
                tokens_list.append(tokens)

            # Scores: colunas 1..n
            for j, score in enumerate(scores):
                result_np[i, 1 + j] = score

            elapsed = time.perf_counter() - start_time

            result_np[i, n_criteria + 1] = round(elapsed, 2)

            # Tokens: colunas após Time
            for j, tokens in enumerate(tokens_list):
                result_np[i, n_criteria + 2 + j] = tokens

            result_np[i, n_criteria + 2 + n_criteria] = sum(tokens_list)

            print(f"Processing #{i+1} of {len(data)} in {elapsed:.2f}s")

        return pd.DataFrame(result_np, columns=columns)
    
    def run_and_export(self, data: list[dict], criteria: dict) -> pd.DataFrame:
        df = self.run(data, criteria)

        filename = f"out/results-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{self.model_name}.xlsx"
        df.to_excel(filename, index=False, engine="openpyxl")
        print(f"File '{filename}' created!")
        return df


