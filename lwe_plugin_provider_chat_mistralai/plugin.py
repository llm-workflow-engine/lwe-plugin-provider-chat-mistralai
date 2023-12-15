# TODO: Fix this and remove mistralai.py when
# https://github.com/langchain-ai/langchain/pull/14775 lands.
# from langchain.chat_models.mistralai import ChatMistralAI
from lwe_plugin_provider_chat_mistralai.mistralai import ChatMistralAI

from lwe.core.provider import Provider, PresetValue

MISTRAL_AI_DEFAULT_MODEL = "mistral-small"


class ProviderChatMistralai(Provider):
    """
    Access to chat Anthropic models
    """

    @property
    def model_property_name(self):
        return "model"

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
            'models': {
                'mistral-tiny': {
                    'max_tokens': 32768,
                },
                'mistral-small': {
                    'max_tokens': 32768,
                },
                'mistral-medium': {
                    'max_tokens': 32768,
                },
            },
        }

    @property
    def default_model(self):
        return MISTRAL_AI_DEFAULT_MODEL

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return ChatMistralAI

    def customization_config(self):
        return {
            'mistral_api_key': PresetValue(str, include_none=True, private=True),
            'endpoint': PresetValue(str, include_none=True),
            'max_retries': PresetValue(int, min_value=1, include_none=True),
            'timeout': PresetValue(int, min_value=1, include_none=True),
            'max_concurrent_requests': PresetValue(int, min_value=1, include_none=True),
            'model': PresetValue(str, options=self.available_models),
            'temperature': PresetValue(float, min_value=0.0, max_value=1.0, include_none=True),
            'max_tokens': PresetValue(int, min_value=1, include_none=True),
            'top_p': PresetValue(float, min_value=0.0, max_value=1.0, include_none=True),
            'random_seed': PresetValue(int, include_none=True),
            'safe_mode': PresetValue(bool, include_none=True),
        }
