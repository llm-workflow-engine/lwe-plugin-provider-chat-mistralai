import os
import requests

from langchain_mistralai.chat_models import ChatMistralAI

from lwe.core.provider import Provider, PresetValue

MISTRAL_AI_DEFAULT_MODEL = "mistral-small"
MISTRAL_AI_API_BASE = "https://api.mistral.ai/v1"


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

    def fetch_models(self):
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
            models_list = models_data.get('data')
            if not models_list:
                raise ValueError('Could not retrieve models')
            models = {model['id']: {'max_tokens': model['max_context_length']} for model in models_list if 'max_context_length' in model and model['max_context_length'] and 'capabilities' in model and 'completion_chat' in model['capabilities'] and model['capabilities']['completion_chat']}
            return models
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Could not retrieve models: {e}")

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
