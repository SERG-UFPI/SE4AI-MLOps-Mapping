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
    Classe que utiliza o consenso entre ChatGPT e Gemini para triagem em SLR.
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
                print(f"Checkpoint encontrado! Retomando a partir do artigo {start_index + 1}...")

        for i in range(start_index, len(articles)):
            article = articles[i]
            print(f"[{i+1}/{len(articles)}] Processando Consenso: {article.get('Título', 'Sem Título')[:50]}...")

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
            prompt = self._build_prompt(article, ic_value)
            system_instr = self._build_system_instruction()

            # 1. Query ChatGPT
            start_gpt = time.perf_counter()
            gpt_response = self.gpt_client.beta.chat.completions.parse(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format=SingleCriterionEvaluation,
            )
            gpt_eval = gpt_response.choices[0].message.parsed
            latency_gpt = time.perf_counter() - start_gpt

            # 2. Query Gemini
            start_gemini = time.perf_counter()
            gemini_response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instr,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=SingleCriterionEvaluation
                ),
            )
            gemini_eval_dict = json.loads(gemini_response.text)
            latency_gemini = time.perf_counter() - start_gemini

            # 3. Consensus Logic
            gpt_decision = gpt_eval.decision
            gemini_decision = gemini_eval_dict["decision"]
            
            final_decision = gpt_decision if gpt_decision == gemini_decision else "UNCLEAR"
            
            print(f"   {ic_key} -> GPT: {gpt_decision} | Gemini: {gemini_decision} | Final: {final_decision}")

            ic_entry = {
                "criterion": ic_key,
                "decision": final_decision,
                "individual_evaluations": {
                    "chatgpt": {
                        "decision": gpt_decision,
                        "reasoning": gpt_eval.reasoning,
                        "latency": round(latency_gpt, 2),
                        "tokens": gpt_response.usage.total_tokens
                    },
                    "gemini": {
                        "decision": gemini_decision,
                        "reasoning": gemini_eval_dict["reasoning"],
                        "latency": round(latency_gemini, 2),
                        "tokens": (gemini_response.usage_metadata.prompt_token_count + (gemini_response.usage_metadata.candidates_token_count or 0))
                    }
                }
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
