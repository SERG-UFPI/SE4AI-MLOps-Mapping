import json
from pathlib import Path
import time
from typing import Dict, Any, List, Literal
from openai import OpenAI
from pydantic import BaseModel, Field
from .base import BaseLLM


class SingleCriterionEvaluation(BaseModel):
    reasoning: str = Field(
        description="Provide EXACTLY ONE short sentence. MAXIMUM 20 WORDS. Include a brief quote and justification."
    )
    decision: Literal["YES", "NO", "UNCLEAR"] = Field(
        description="YES: Explicit evidence meets the criterion. NO: Fails or violates the criterion. UNCLEAR: Vague or lacks details."
    )


class ChatGPTLLM(BaseLLM):
    """
    Classe para automação de triagem em SLR utilizando ChatGPT (OpenAI), com foco em avaliações ternárias (YES/NO/UNCLEAR).
    """

    def __init__(self, api_key: str, model_name: str, config: Dict[str, Any]):
        """
        Inicializa o cliente da API e define o modelo.
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.config = config

    def batch_classify(
        self, articles: List[Dict[str, Any]], criteria: Dict[str, str], checkpoint_interval: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Processa artigos em lote com salvamento parcial e suporte a retomar progresso.
        """
        checkpoint_path = Path("experiments/tmp_checkpoint_chatgpt.json")
        results = []
        start_index = 0

        # 1. Tenta retomar de um checkpoint anterior
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                start_index = len(results)
                print(f"Checkpoint encontrado! Retomando a partir do artigo {start_index + 1}...")

        # 2. Processa apenas os artigos restantes
        for i in range(start_index, len(articles)):
            article = articles[i]
            print(f"[{i+1}/{len(articles)}] Processando (ChatGPT): {article.get('Título', 'Sem Título')[:50]}...")

            try:
                result = self.evaluate_article(article, criteria)
                results.append(result)

                # Respeita limites de taxa se necessário
                time.sleep(1)

                if (i + 1) % checkpoint_interval == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    print(f"Progresso guardado (Artigo {i+1})")

            except Exception as e:
                print(f"Erro crítico no artigo {i+1}: {e}")
                print("O progresso atual foi mantido no checkpoint. Podes reiniciar o script.")
                break

        # 4. Limpeza: Se terminou tudo, apaga o checkpoint temporário
        if len(results) == len(articles) and checkpoint_path.exists():
            checkpoint_path.unlink()
            print("✅ Processamento completo. Ficheiro temporário removido.")

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

            prompt = self._build_prompt(article, ic_value)

            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._build_system_instruction()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.get("temperature", 0.0),
                max_completion_tokens=self.config.get("max_tokens") or self.config.get("max_output_tokens", 500),
                top_p=self.config.get("top_p", 1.0),
                response_format=SingleCriterionEvaluation,
            )

            end_ic = time.perf_counter()
            latency = end_ic - start_ic
            usage = response.usage

            evaluation_result = response.choices[0].message.parsed

            if not evaluation_result:
                raise ValueError("Falha ao analisar a resposta estruturada do ChatGPT.")

            print(f"IC: {ic_key} | Decision: {evaluation_result.decision}")

            ic_entry = {
                "criterion": ic_key,
                "decision": evaluation_result.decision,
                "reasoning": evaluation_result.reasoning,
                "telemetry": {
                    "latency_sec": round(latency, 2),
                    "tokens_prompt": usage.prompt_tokens,
                    "tokens_completion": usage.completion_tokens,
                },
            }

            inclusion_details.append(ic_entry)

        return {
            "article_metadata": {
                "title": article.get("Título"),
                "author": article.get("Autor"),
                "year": article.get("Ano"),
                "link": article.get("Link do pdf"),
                "DOI": article.get("DOI"),
                "ISBNs": article.get("ISBNs"),
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

    def _build_prompt(self, article: Dict[str, Any], criterion: str) -> str:
        title = article.get("Título")
        abstract = article.get("Abstract")

        return (
            "STUDY DATA:\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n\n"
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
