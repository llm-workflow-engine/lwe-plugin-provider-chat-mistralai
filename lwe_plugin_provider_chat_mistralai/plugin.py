from langchain_mistralai.chat_models import ChatMistralAI

from lwe.core.provider import Provider, PresetValue

MISTRAL_AI_DEFAULT_MODEL = "mistral-small"


class CustomChatMistralAI(ChatMistralAI):

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_mistralai"


class ProviderChatMistralai(Provider):
    """
    Access to chat MistralAI models
    """

    @property
    def model_property_name(self):
        return "model"

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
        }

    @property
    def default_model(self):
        return MISTRAL_AI_DEFAULT_MODEL

    @property
    def static_models(self):
        """To get the latest list of models:
        curl -X GET "https://api.mistral.ai/v1/models" -H "Authorization: Bearer $MISTRAL_API_KEY" -H "Content-Type: application/json" | jq
        """
        return {
            'open-mistral-7b': {
                'max_tokens': 32768,
            },
            'open-mixtral-8x7b': {
                'max_tokens': 32768,
            },
            'open-mixtral-8x22b': {
                'max_tokens': 65536,
            },
            'mistral-tiny': {
                'max_tokens': 32768,
            },
            'mistral-small': {
                'max_tokens': 32768,
            },
            'mistral-small-latest': {
                'max_tokens': 32768,
            },
            'mistral-medium': {
                'max_tokens': 32768,
            },
            'mistral-medium-latest': {
                'max_tokens': 32768,
            },
            'mistral-large-latest': {
                'max_tokens': 131072,
            },
            'codestral-latest': {
                'max_tokens': 32768,
            },
            'open-codestral-mamba': {
                'max_tokens': 262144,
            },
            'open-mistral-nemo': {
                'max_tokens': 131072,
            },
        }

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatMistralAI

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
            "tools": None,
            "tool_choice": None,
        }
