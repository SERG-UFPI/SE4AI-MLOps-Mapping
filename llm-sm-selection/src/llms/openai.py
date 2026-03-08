# src/llms/openai.py
import json
from typing import Any, Dict
import openai
from src.llms.base import BaseLLM

class OpenAILLM(BaseLLM):
    """
    Wrapper for the OpenAI API.
    """

    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        openai.api_key = self.api_key

    def screen_article(self, article: Dict[str, Any], criteria_prompt: str) -> Dict[str, Any]:
        """
        Screens an article using the OpenAI API.
        """
        prompt = self._create_prompt(article, criteria_prompt)
        
        try:
            # Note: The 'openai.ChatCompletion.create' call is mocked for this example.
            # In a real scenario, you would make an actual API call.
            # response = openai.ChatCompletion.create(
            #     model=self.model,
            #     messages=[{"role": "user", "content": prompt}],
            #     response_format={"type": "json_object"},
            # )
            # response_content = response.choices[0].message.content
            
            # Mocking the response for demonstration purposes
            mock_response_content = json.dumps({
                "decision": "include",
                "reason": "The article is about software engineering and machine learning."
            })
            response_content = mock_response_content

            result = json.loads(response_content)
            
            return {
                "id": article.get("id"),
                "decision": result.get("decision", "error"),
                "reason": result.get("reason", "Could not parse response."),
                "llm_used": self.model,
            }

        except Exception as e:
            print(f"An error occurred with OpenAI model {self.model}: {e}")
            return {
                "id": article.get("id"),
                "decision": "error",
                "reason": str(e),
                "llm_used": self.model,
            }
