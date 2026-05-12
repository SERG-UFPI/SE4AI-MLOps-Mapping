import json
from pathlib import Path
import time
from typing import Dict, Any, List
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from .base import BaseLLM

class SingleCriterionNumericEvaluation(BaseModel):
    reasoning: str = Field(
        description="Provide EXACTLY ONE short sentence. MAXIMUM 20 WORDS. Include a brief quote and justification."
    )
    score: int = Field(
        description="Numeric score from 1 to 7 based on the relevance to the criterion.",
        ge=1,
        le=7
    )

class GeminiNumericLLM(BaseLLM):
    """
    Classe para automação de triagem em SLR utilizando Gemini com escala numérica de 1 a 7.
    """

    def __init__(self, api_key: str, model_name: str, config: Dict[str, Any]):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.config = config

    def batch_classify(
        self, articles: List[Dict[str, Any]], criteria: Dict[str, str], checkpoint_interval: int = 10
    ) -> List[Dict[str, Any]]:
        checkpoint_path = Path("experiments/tmp_checkpoint_gemini_numeric.json")
        results = []
        start_index = 0

        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                start_index = len(results)
                print(f"Checkpoint encontrado! Retomando a partir do artigo {start_index + 1}...")

        for i in range(start_index, len(articles)):
            article = articles[i]
            print(f"[{i+1}/{len(articles)}] Processando (Gemini Numeric): {article.get('Título', 'Sem Título')[:50]}...")
            
            try:
                result = self.evaluate_article(article, criteria)
                results.append(result)
                #time.sleep(2) 
                
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
                    response_schema=SingleCriterionNumericEvaluation
                ),
            )

            end_ic = time.perf_counter()
            evaluation_result = json.loads(response.text)

            ic_entry = {
                "criterion": ic_key,
                "score": evaluation_result["score"],
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
            "CONTEXT: You are screening articles for a Systematic Literature Review (SLR) "
            "on software engineering practices applied to robotic systems using ROS "
            "(Robot Operating System).\n\n"
            "STUDY DATA:\n"
            f"Title: {article.get('Título')}\n"
            f"Abstract: {article.get('Abstract')}\n\n"
            "CRITERION TO EVALUATE:\n"
            f"{criterion}"
        )

    def _build_system_instruction(self) -> str:
        return (
            "You are a software engineering researcher screening studies for a Systematic "
            "Literature Review (SLR). Evaluate whether the study's title and abstract meet "
            "a given inclusion criterion, using a relevance scale from 1 to 7.\n\n"
            "SCORING SCALE:\n"
            "1 - Clearly does NOT meet the criterion (strong evidence of exclusion)\n"
            "2 - Very unlikely to meet the criterion\n"
            "3 - Unlikely to meet the criterion\n"
            "4 - Uncertain — abstract provides insufficient information to decide\n"
            "5 - Likely meets the criterion\n"
            "6 - Very likely meets the criterion\n"
            "7 - Clearly MEETS the criterion (strong evidence of inclusion)\n\n"
            "EVALUATION RULES:\n"
            "1. Consider both explicit statements AND strong contextual implications in the title/abstract.\n"
            "2. If the abstract is short or vague but the title strongly implies the criterion is met, score 4-5.\n"
            "3. Reserve score 1 only when the abstract clearly shows the criterion is NOT met.\n"
            "4. Reserve score 7 only when the abstract clearly confirms the criterion IS met.\n"
            "5. STRICT LENGTH LIMIT: Your 'reasoning' MUST be absolutely under 20 words. Be telegraphic and direct."
        )
