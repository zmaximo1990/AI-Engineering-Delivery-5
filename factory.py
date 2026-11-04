import logging
from typing import Optional

from llm import AnthropicClient, BaseLLMClient, GeminiClient, OpenAIClient
from schemas import LLMConfig, LLMProvider
from settings import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory that builds an LLM client from an LLMConfig."""

    @staticmethod
    def create_client(config: LLMConfig) -> BaseLLMClient:
        """Build the client that matches ``config.provider``."""
        logger.info("Creating LLM client for provider=%s", config.provider.value)

        kwargs = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        client: Optional[BaseLLMClient] = None
        if config.provider == LLMProvider.OPENAI:
            client = OpenAIClient(**kwargs)
        elif config.provider == LLMProvider.ANTHROPIC:
            client = AnthropicClient(**kwargs)
        elif config.provider == LLMProvider.GEMINI:
            # Gemini spends part of the budget on thinking tokens.
            kwargs["max_tokens"] = max(config.max_tokens, settings.gemini_max_tokens)
            client = GeminiClient(**kwargs)
        else:
            raise ValueError(f"Invalid provider: {config.provider}")
        logger.info("LLM client created: %s", type(client).__name__)
        return client
