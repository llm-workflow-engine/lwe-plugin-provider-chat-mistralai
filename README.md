# LLM Workflow Engine (LWE) Chat MistralAI Provider plugin

Chat MistralAI Provider plugin for [LLM Workflow Engine](https://github.com/llm-workflow-engine/llm-workflow-engine)

Access to [MistralAI](https://docs.mistral.ai) chat models.

## Installation

### Export API key

Grab a MistralAI API key from [https://console.mistral.ai/users/api-keys](https://console.mistral.ai/users/api-keys)

Export the key into your local environment:

```bash
export MISTRAL_API_KEY=<API_KEY>
```

### From packages

Install the latest version of this software directly from github with pip:

```bash
pip install git+https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-mistralai
```

### From source (recommended for development)

Install the latest version of this software directly from git:

```bash
git clone https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-mistralai.git
```

Install the development package:

```bash
cd lwe-plugin-provider-chat-mistralai
pip install -e .
```

## Configuration

Add the following to `config.yaml` in your profile:

```yaml
plugins:
  enabled:
    - provider_chat_mistralai
    # Any other plugins you want enabled...
```

## Usage

From a running LWE shell:

```
/provider chat_mistralai
/model model mistral-small
```
