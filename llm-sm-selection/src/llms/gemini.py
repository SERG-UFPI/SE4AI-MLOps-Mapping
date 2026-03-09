import time
from typing import Dict, Any, List
from google import genai
from google.genai import types
from .base import BaseLLM


class GeminiLLM(BaseLLM):
    """
    Classe para automação de triagem em SLR utilizando Gemini
    """

    def __init__(self, api_key: str, model_name: str, config: Dict[str, Any]):
        """
        Inicializa o cliente da API e define o modelo.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.config = config

    def batch_classify(
        self, articles: List[Dict[str, Any]], criteria: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Processa uma lista de artigos em lote."""
        results = []
        for article in articles:
            print(f"Processando: {article['Título'][:50]}...")
            results.append(self.evaluate_article(article, criteria))
        return results

    def evaluate_article(
        self, article: Dict[str, Any], criteria_list: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Avalia um artigo individualmente contra uma lista de critérios.
        Retorna telemetria detalhada por IC e o total acumulado.
        """
        inclusion_details = []

        for ic_key, ic_value in criteria_list.items():
            start_ic = time.perf_counter()

            prompt = f"{ic_value}"

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_instruction(article),
                    temperature=self.config["temperature"],
                    max_output_tokens=self.config["max_output_tokens"],
                    top_p=self.config["top_p"]
                ),
            )

            end_ic = time.perf_counter()
            latency = end_ic - start_ic
            usage = response.usage_metadata

            try:
                print(f"Resposta bruta para {ic_key}: '{response.text.strip()}'")
                score = int(response.text.strip())
            except ValueError:
                score = None

            ic_entry = {
                "criterion": ic_key,
                "score": score,
                "telemetry": {
                    "latency_sec": round(latency, 2),
                    "tokens_prompt": usage.prompt_token_count,
                    "tokens_completion": usage.candidates_token_count,
                },
            }

            inclusion_details.append(ic_entry)

        return {
            "article_metadata": {
                "title": article.get("title"),
                "author": article.get("author"),
                "year": article.get("year"),
                "DOI": article.get("DOI"),
            },
            "inclusion_results": inclusion_details,
            "total_article_telemetry": self._summarize_telemetry(inclusion_details),
        }

    def _summarize_telemetry(self, details: List[Dict]) -> Dict:
        return {
            "total_latency": round(
                sum(d["telemetry"]["latency_sec"] for d in details), 2
            ),
            "total_tokens": sum(
                d["telemetry"]["tokens_prompt"] + d["telemetry"]["tokens_completion"]
                for d in details
            ),
            "model": self.model_name,
        }

    def _build_system_instruction(self, item: dict) -> str:
        likert_scale = (
            "1 - Strongly disagree, 2 - Disagree, 3 - Somewhat disagree, "
            "4 - Neither agree nor disagree, 5 - Somewhat agree, 6 - Agree, "
            "and 7 - Strongly agree"
        )

        return (
            "Assume you are a software engineering researcher conducting a systematic literature review (SLR). "
            "Your goal is to evaluate a primary study based on its title and abstract.\n\n"
            "STUDY DATA:\n"
            f"Title: {item.get('Título', 'N/A')}\n"
            f"Abstract: {item.get('Abstract', 'N/A')}\n\n"
            "EVALUATION RULES:\n"
            f"1. Use a 1-7 Likert scale: ({likert_scale}).\n"
            "2. Rate your agreement with the criteria provided in the prompt.\n"
            "3. Output ONLY the number corresponding to your rating."
        )
