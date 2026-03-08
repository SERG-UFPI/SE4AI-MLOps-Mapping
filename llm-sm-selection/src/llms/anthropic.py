# src/llms/anthropic.py
import json
from typing import Any, Dict
import anthropic
from src.llms.base import BaseLLM

class AnthropicLLM(BaseLLM):
    """
    Wrapper for the Anthropic API.
    """

    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def screen_article(self, article: Dict[str, Any], criteria_prompt: str) -> Dict[str, Any]:
        """
        Screens an article using the Anthropic API.
        """
        prompt = self._create_prompt(article, criteria_prompt)
        
        try:
            # Note: The 'client.messages.create' call is mocked for this example.
            # response = self.client.messages.create(
            #     model=self.model,
            #     max_tokens=1024,
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": prompt,
            #         }
            #     ],
            # )
            # response_content = response.content[0].text

            # Mocking the response for demonstration purposes
            mock_response_content = json.dumps({
                "decision": "exclude",
                "reason": "The study is a systematic review, which is an exclusion criterion."
            })
            response_content = mock_response_content
            
            # Extract JSON from a potentially larger string
            start_index = response_content.find('{')
            end_index = response_content.rfind('}') + 1
            json_str = response_content[start_index:end_index]
            
            result = json.loads(json_str)

            return {
                "id": article.get("id"),
                "decision": result.get("decision", "error"),
                "reason": result.get("reason", "Could not parse response."),
                "llm_used": self.model,
            }

        except Exception as e:
            print(f"An error occurred with Anthropic model {self.model}: {e}")
            return {
                "id": article.get("id"),
                "decision": "error",
                "reason": str(e),
                "llm_used": self.model,
            }
