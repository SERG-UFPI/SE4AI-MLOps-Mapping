import json
from pathlib import Path
import time
from typing import Dict, Any, List, Literal
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from .base import BaseLLM

class SingleCriterionEvaluation(BaseModel):
    reasoning: str = Field(
        description="Provide EXACTLY ONE short sentence. MAXIMUM 20 WORDS. Include a brief quote and justification."
    )
    decision: Literal["YES", "NO", "UNCLEAR"] = Field(
        description="YES: Explicit evidence meets the criterion. NO: Fails or violates the criterion. UNCLEAR: Vague or lacks details."
    )

class GeminiLLMV2(BaseLLM):
    """
    Classe para automação de triagem em SLR utilizando Gemini.
    """

    def __init__(self, api_key: str, model_name: str, config: Dict[str, Any]):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.config = config

    def batch_classify(
        self, articles: List[Dict[str, Any]], criteria: Dict[str, str], checkpoint_interval: int = 10
    ) -> List[Dict[str, Any]]:
        checkpoint_path = Path("experiments/tmp_checkpoint_gemini.json")
        results = []
        start_index = 0

        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                start_index = len(results)
                print(f"Checkpoint encontrado! Retomando a partir do artigo {start_index + 1}...")

        for i in range(start_index, len(articles)):
            article = articles[i]
            print(f"[{i+1}/{len(articles)}] Processando (Gemini): {article.get('Título', 'Sem Título')[:50]}...")
            
            try:
                result = self.evaluate_article(article, criteria)
                results.append(result)
                time.sleep(2) 
                
                if (i + 1) % checkpoint_interval == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)

            except Exception as e:
                print(f"Erro no artigo {i+1}: {e}")
                break

        if len(results) == len(articles) and checkpoint_path.exists():
            checkpoint_path.unlink()

        return results

    def evaluate_article(
        self, article: Dict[str, Any], criteria_list: Dict[str, str]
    ) -> Dict[str, Any]:
        inclusion_details = []

        for ic_key, ic_value in criteria_list.items():
            start_ic = time.perf_counter()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self._build_prompt(article, ic_value),
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_instruction(),
                    temperature=self.config.get("temperature", 0.0),
                    max_output_tokens=self.config.get("max_output_tokens", 500),
                    top_p=self.config.get("top_p", 1.0),
                    response_mime_type="application/json",
                    response_schema=SingleCriterionEvaluation
                ),
            )

            end_ic = time.perf_counter()
            evaluation_result = json.loads(response.text)

            ic_entry = {
                "criterion": ic_key,
                "decision": evaluation_result["decision"],
                "reasoning": evaluation_result["reasoning"],
                "telemetry": {
                    "latency_sec": round(end_ic - start_ic, 2),
                    "tokens_prompt": response.usage_metadata.prompt_token_count,
                    "tokens_completion": response.usage_metadata.candidates_token_count or 0,
                },
            }
            inclusion_details.append(ic_entry)

        return {
            "article_metadata": {
                "title": article.get("Título"),
                "author": article.get("Autor"),
                "year": article.get("Ano")
            },
            "inclusion_results": inclusion_details
        }
    
    def _build_prompt(self, article: Dict[str, Any], criterion: str) -> str:
        return (
            "STUDY DATA:\n"
            f"Title: {article.get('Título')}\n"
            f"Abstract: {article.get('Abstract')}\n\n"
            "CRITERION TO EVALUATE:\n"
            f"{criterion}"
        )

    def _build_system_instruction(self) -> str:
        return (
            "Assume you are a strict software engineering researcher conducting a "
            "systematic literature review (SLR). Your goal is to evaluate a primary "
            "study based on its title and abstract against a specific criterion.\n\n"
            "EVALUATION RULES:\n"
            "1. Evaluate if the study meets the criterion provided in the prompt.\n"
            "2. Anchor your reasoning explicitly in the text of the abstract.\n"
            "3. Be extremely strict. Do not assume information that is not explicitly written.\n"
            "4. STRICT LENGTH LIMIT: Your 'reasoning' MUST be absolutely under 20 words. Be telegraphic and direct."
        )
