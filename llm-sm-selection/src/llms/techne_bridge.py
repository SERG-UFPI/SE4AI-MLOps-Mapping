import json
from pathlib import Path
import time
from typing import Dict, Any, List, Literal
import requests
from .base import BaseLLM

class TechneBridgeLLM(BaseLLM):
    """
    Classe para automação de triagem em SLR utilizando a API TECHNE HPC Bridge.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.api_key = "techne2026"
        self.model_name = model_name
        self.config = config
        self.base_url = config.get("base_url", "http://10.94.80.13:8080").rstrip("/")

    def batch_classify(
        self, articles: List[Dict[str, Any]], criteria: Dict[str, str], checkpoint_interval: int = 10
    ) -> List[Dict[str, Any]]:
        checkpoint_path = Path("experiments/tmp_checkpoint_techne.json")
        results = []
        start_index = 0

        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                start_index = len(results)
                print(f"Checkpoint encontrado! Retomando a partir do artigo {start_index + 1}...")

        for i in range(start_index, len(articles)):
            article = articles[i]
            print(f"[{i+1}/{len(articles)}] Processando (Techne Bridge): {article.get('Título', 'Sem Título')[:50]}...")

            try:
                result = self.evaluate_article(article, criteria)
                results.append(result)
                time.sleep(1)

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

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self._build_system_instruction()},
                    {"role": "user", "content": self._build_prompt(article, ic_value)},
                ],
                "stream": False
            }

            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }

            response = requests.post(
                f"{self.base_url}/v1/chat",
                json=payload,
                headers=headers,
                timeout=self.config.get("timeout", 120)
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            end_ic = time.perf_counter()
            
            # O bridge retorna a resposta bruta do Ollama
            content = response_json.get("message", {}).get("content", "")
            
            # Tenta extrair JSON do conteúdo caso o modelo retorne markdown ou texto extra
            evaluation_result = self._parse_json_response(content)

            ic_entry = {
                "criterion": ic_key,
                "decision": evaluation_result.get("decision", "UNCLEAR"),
                "reasoning": evaluation_result.get("reasoning", "Failed to parse response."),
                "telemetry": {
                    "latency_sec": round(end_ic - start_ic, 2),
                    "tokens_prompt": response_json.get("prompt_eval_count", 0),
                    "tokens_completion": response_json.get("eval_count", 0),
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

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        try:
            # Tenta o parse direto
            return json.loads(content)
        except json.JSONDecodeError:
            # Tenta extrair entre blocos de código markdown se existirem
            import re
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Se falhar, tenta encontrar o primeiro { e o último }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(content[start:end+1])
                except json.JSONDecodeError:
                    pass
            
            return {"decision": "UNCLEAR", "reasoning": f"Erro ao decodificar JSON. Conteúdo bruto: {content[:100]}..."}

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
            "4. STRICT LENGTH LIMIT: Your 'reasoning' MUST be absolutely under 20 words. Be telegraphic and direct.\n\n"
            "OUTPUT FORMAT:\n"
            "You MUST respond ONLY with a JSON object in the following format:\n"
            "{\n"
            '  "reasoning": "ONE sentence, max 20 words, with quote.",\n'
            '  "decision": "YES" | "NO" | "UNCLEAR"\n'
            "}"
        )
