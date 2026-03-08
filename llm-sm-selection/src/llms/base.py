# src/llms/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model wrappers.
    
    Each LLM provider should implement this interface to ensure
    compatibility with the screening pipeline.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def screen_article(self, article: Dict[str, Any], criteria_prompt: str) -> Dict[str, Any]:
        """
        Screens a single article against the given criteria.

        Args:
            article: A dictionary representing the article to be screened.
                     Expected to have 'id', 'title', and 'abstract'.
            criteria_prompt: A formatted string of the screening criteria.

        Returns:
            A dictionary containing:
            - 'id': The article's ID.
            - 'decision': 'include' or 'exclude'.
            - 'reason': A brief justification for the decision.
            - 'llm_used': The name of the model used for screening.
        """
        pass

    def _create_prompt(self, article: Dict[str, Any], criteria_prompt: str) -> str:
        """
        Creates the full prompt to be sent to the LLM.
        """
        title = article.get("title", "No Title")
        abstract = article.get("abstract", "No Abstract")

        prompt = (
            f"{criteria_prompt}
"
            f"Article Title: {title}
"
            f"Article Abstract: {abstract}

"
            "Based on the title and abstract, should this article be included or excluded? "
            "Provide your decision and a brief reason.
"
            "Respond in JSON format with keys: 'decision' ('include' or 'exclude') and 'reason'."
        )
        return prompt
