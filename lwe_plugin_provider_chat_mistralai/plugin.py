"""LWE chat mistralai provider integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import os

from langchain_core.messages import BaseMessage
from langchain_mistralai.chat_models import ChatMistralAI
from typing_extensions import override
import requests

from lwe.core.provider import PresetValue, Provider

MISTRAL_AI_DEFAULT_MODEL = "mistral-small"
MISTRAL_AI_API_BASE = "https://api.mistral.ai/v1"


class CustomChatMistralAI(ChatMistralAI):

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat_mistralai"


class ProviderChatMistralai(Provider):
    """
    Access to chat MistralAI models
    """

    @property
    @override
    def model_property_name(self) -> str:
        return "model"

    @property
    @override
    def capabilities(self) -> dict[str, bool]:
        return {
            "chat": True,
            "validate_models": True,
        }

    @property
    @override
    def default_model(self) -> str:
        return MISTRAL_AI_DEFAULT_MODEL

    def fetch_models(self) -> dict[str, dict[str, int | None]]:
        models_url = f"{MISTRAL_AI_API_BASE}/models"
        try:
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
            }
            response = requests.get(models_url, headers=headers)
            response.raise_for_status()
            models_data = response.json()
            models_list = models_data.get("data")
            if not models_list:
                raise ValueError("Could not retrieve models")
            models = {
                model["id"]: {"max_tokens": model["max_context_length"]}
                for model in models_list
                if "max_context_length" in model
                and model["max_context_length"]
                and "capabilities" in model
                and "completion_chat" in model["capabilities"]
                and model["capabilities"]["completion_chat"]
            }
            return models
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Could not retrieve models: {e}")

    @override
    def prepare_messages_method(
        self,
    ) -> Callable[[list[dict[str, Any]]], list[BaseMessage]]:
        return self.prepare_messages_for_llm_chat

    @override
    def llm_factory(self) -> type[CustomChatMistralAI]:
        return CustomChatMistralAI

    @override
    def customization_config(self) -> dict[str, Any]:
        return {
            "mistral_api_key": PresetValue(str, include_none=True, private=True),
            "endpoint": PresetValue(str, include_none=True),
            "max_retries": PresetValue(int, min_value=1, include_none=True),
            "timeout": PresetValue(int, min_value=1, include_none=True),
            "max_concurrent_requests": PresetValue(int, min_value=1, include_none=True),
            "model": PresetValue(str, options=self.available_models),
            "temperature": PresetValue(
                float, min_value=0.0, max_value=1.0, include_none=True
            ),
            "max_tokens": PresetValue(int, min_value=1, include_none=True),
            "top_p": PresetValue(
                float, min_value=0.0, max_value=1.0, include_none=True
            ),
            "random_seed": PresetValue(int, include_none=True),
            "safe_mode": PresetValue(bool, include_none=True),
            "tools": None,
            "tool_choice": None,
        }
