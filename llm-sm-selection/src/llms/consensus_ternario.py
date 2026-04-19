import json
from pathlib import Path
import time
from typing import Dict, Any, List, Literal
from openai import OpenAI
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

class ConsensusLLM(BaseLLM):
    """
    Classe que utiliza o consenso entre ChatGPT e Gemini para triagem.
    """

    def __init__(self, openai_key: str, gemini_key: str, gpt_model: str, gemini_model: str, config: Dict[str, Any]):
        self.gpt_client = OpenAI(api_key=openai_key)
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.gpt_model = gpt_model
        self.gemini_model = gemini_model
        self.config = config

    def batch_classify(
        self, articles: List[Dict[str, Any]], criteria: Dict[str, str], checkpoint_interval: int = 10
    ) -> List[Dict[str, Any]]:
        checkpoint_path = Path("experiments/tmp_checkpoint_consensus.json")
        results = []
        start_index = 0

        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                start_index = len(results)
                print(f"Retomando a partir do artigo {start_index + 1}...")

        for i in range(start_index, len(articles)):
            article = articles[i]
            print(f"[{i+1}/{len(articles)}] Processando Consenso: {article.get('Título', 'Sem Título')[:50]}...")

            try:
                result = self.evaluate_article(article, criteria)
                results.append(result)

                # Delay para respeitar limites de taxa (ajustável)
                time.sleep(2)

                if (i + 1) % checkpoint_interval == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    print(f"Progresso guardado (Artigo {i+1})")

            except Exception as e:
                print(f"Erro crítico no artigo {i+1}: {e}")
                break

        if len(results) == len(articles) and checkpoint_path.exists():
            checkpoint_path.unlink()

        return results

    def evaluate_article(
        self, article: Dict[str, Any], criteria_list: Dict[str, str]
    ) -> Dict[str, Any]:
        inclusion_details = []

        for ic_key, ic_value in criteria_list.items():
            # 1. Query ChatGPT
            gpt_eval = self._query_gpt(article, ic_value)
            
            # 2. Query Gemini
            gemini_eval = self._query_gemini(article, ic_value)

            # 3. Consensus Logic
            # Se ambos concordarem, mantemos a decisão.
            # Se divergirem, marcamos como UNCLEAR para indicar conflito de IA.
            if gpt_eval['decision'] == gemini_eval['decision']:
                final_decision = gpt_eval['decision']
            else:
                final_decision = "UNCLEAR"

            print(f"   {ic_key} -> GPT: {gpt_eval['decision']} | Gemini: {gemini_eval['decision']} | Final: {final_decision}")

            ic_entry = {
                "criterion": ic_key,
                "decision": final_decision,
                "gpt_reasoning": gpt_eval['reasoning'],
                "gemini_reasoning": gemini_eval['reasoning'],
                "telemetry": {
                    "gpt_latency": gpt_eval['latency'],
                    "gemini_latency": gemini_eval['latency']
                }
            }
            inclusion_details.append(ic_entry)

        return {
            "article_metadata": {
                "title": article.get("Título"),
                "author": article.get("Autor"),
                "year": article.get("Ano")
            },
            "inclusion_results": inclusion_details,
            "total_article_telemetry": {
                "model_gpt": self.gpt_model,
                "model_gemini": self.gemini_model
            }
        }

    def _query_gpt(self, article: Dict[str, Any], criterion: str) -> Dict:
        start = time.perf_counter()
        response = self.gpt_client.beta.chat.completions.parse(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": self._build_system_instruction()},
                {"role": "user", "content": self._build_prompt(article, criterion)},
            ],
            temperature=0.0,
            response_format=SingleCriterionEvaluation,
        )
        latency = time.perf_counter() - start
        parsed = response.choices[0].message.parsed
        return {
            "decision": parsed.decision,
            "reasoning": parsed.reasoning,
            "latency": round(latency, 2)
        }

    def _query_gemini(self, article: Dict[str, Any], criterion: str) -> Dict:
        start = time.perf_counter()
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=self._build_prompt(article, criterion),
            config=types.GenerateContentConfig(
                system_instruction=self._build_system_instruction(),
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=SingleCriterionEvaluation
            ),
        )
        latency = time.perf_counter() - start
        parsed = json.loads(response.text)
        return {
            "decision": parsed["decision"],
            "reasoning": parsed["reasoning"],
            "latency": round(latency, 2)
        }

    def _build_prompt(self, article: Dict[str, Any], criterion: str) -> str:
        return f"STUDY DATA:\nTitle: {article.get('Título')}\nAbstract: {article.get('Abstract')}\n\nCRITERION:\n{criterion}"

    def _build_system_instruction(self) -> str:
        return (
            "Assume you are a strict software engineering researcher. Evaluate if the study meets the criterion.\n"
            "REASONING: Exactly 1 sentence, max 20 words.\n"
            "DECISIONS: YES, NO, or UNCLEAR."
        )
