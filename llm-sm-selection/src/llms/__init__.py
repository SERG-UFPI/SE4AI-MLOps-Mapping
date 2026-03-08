# src/llms/__init__.py
from src.llms.base import BaseLLM
from src.llms.openai import OpenAILLM
from src.llms.anthropic import AnthropicLLM
from src.llms.google import GoogleLLM
from typing import Type

def get_llm_provider(provider_name: str) -> Type[BaseLLM]:
    """
    Factory function to get the LLM provider class.
    """
    provider_map = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "google": GoogleLLM,
    }
    provider = provider_map.get(provider_name.lower())
    if not provider:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return provider
