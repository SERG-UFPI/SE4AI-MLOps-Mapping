# src/llms/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model wrappers.

    Each LLM provider should implement this interface to ensure
    compatibility with the screening pipeline.
    """

    @abstractmethod
    def batch_classify(
        self, articles: List[Dict[str, Any]], criteria: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Processa uma lista de artigos em lote."""
        results = []
        for article in articles:
            print(f"Processando: {article.get('title', 'Sem Título')[:50]}...")
            results.append(self.evaluate_article(article, criteria))
        return results
