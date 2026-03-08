# src/llms/google.py
import json
from typing import Any, Dict
import google.generativeai as genai
from src.llms.base import BaseLLM

class GoogleLLM(BaseLLM):
    """
    Wrapper for the Google Generative AI API (Gemini).
    """

    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def screen_article(self, article: Dict[str, Any], criteria_prompt: str) -> Dict[str, Any]:
        """
        Screens an article using the Google Gemini API.
        """
        prompt = self._create_prompt(article, criteria_prompt)
        
        try:
            # Note: The 'self.client.generate_content' call is mocked.
            # response = self.client.generate_content(prompt)
            # response_content = response.text

            # Mocking the response for demonstration purposes
            mock_response_content = json.dumps({
                "decision": "include",
                "reason": "The article clearly discusses machine learning applications in software engineering."
            })
            response_content = mock_response_content

            # The response from Gemini might include markdown backticks for JSON
            if response_content.startswith("```json"):
                response_content = response_content[7:-4]
            
            result = json.loads(response_content)

            return {
                "id": article.get("id"),
                "decision": result.get("decision", "error"),
                "reason": result.get("reason", "Could not parse response."),
                "llm_used": self.model,
            }

        except Exception as e:
            print(f"An error occurred with Google model {self.model}: {e}")
            return {
                "id": article.get("id"),
                "decision": "error",
                "reason": str(e),
                "llm_used": self.model,
            }
