## Understanding LLM APIs

Let's talk about LLM APIs.

There are a few methods to use the API. One is directly sending HTTP requests, and the other way is to pip install the official OpenAI package, which is up to date with the latest features.

```
pip install openai
```

As the API format is so popular, a lot of other providers also provide theirs in the same format. This is what we call "OpenAI compatible."

## List of Providers that have OpenAI-compatible APIs

Here is a list of providers that are compatible with the OpenAI API, allowing developers to utilize their services in a similar manner to OpenAI's offerings:

- **Groq**

  - Groq provides an API that is mostly compatible with OpenAI's client libraries. Users can configure their applications to run on Groq by changing the `base_url` and using a Groq API key.

- **Mistral AI**

  - Mistral offers an API that supports OpenAI-compatible requests. Developers can access various models through this service.

- **Hugging Face**

  - Hugging Face provides access to numerous models via an API that can be configured similarly to OpenAI's. It is well-known for its extensive model library.

- **Google Vertex AI**

  - Google Vertex AI allows users to interact with large language models in a manner consistent with OpenAI's API.

- **Microsoft Azure OpenAI**

  - Microsoft provides access to OpenAI models through its Azure platform, enabling integration with existing Azure services.

- **Anthropic**
  - Anthropic's models can also be accessed through an API that mimics OpenAI's structure, allowing for similar interactions.

These providers enable developers to leverage different AI capabilities while maintaining compatibility with the familiar OpenAI API structure, facilitating easier integration into applications and workflows.

Some other providers have a different API schema and type, but we have client libraries like [litellm](https://github.com/BerriAI/litellm) that provide a uniform client package for different types of LLMs.

Also, gateways such as [Portkey](https://github.com/Portkey-AI/gateway) provide an OpenAI-compatible API for any of your LLM providers.

In this blog, you are going to learn about the different parameters that go into the API.

### Key Parameters

**Temperature**

- **Description**: The temperature parameter controls the randomness of the model's output. It ranges from 0 to 2.
- **Effects**:

  - **Low Values (0-0.3)**: Outputs are more deterministic and focused, often repeating similar responses.
  - **Medium Values (0.4-0.7)**: Balances creativity and coherence.
  - **High Values (0.8-2)**: Produces more varied and creative outputs but can lead to nonsensical results.

- **Example**:
  ```python
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": "Complete the sentence: 'The key to happiness is'."}],
      temperature=1
  )
  ```

**Top_p**

- **Description**: This parameter implements nucleus sampling, where the model considers only the top `p` probability mass for generating responses.
- **Range**: Between 0 and 1, where lower values restrict the output to more probable tokens.

- **Example**:
  ```python
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": "Write a poem."}],
      top_p=0.9
  )
  ```

**Max Tokens**

- **Description**: Defines the maximum number of tokens (words or parts of words) in the generated response.
- **Default**: Typically set to a maximum of 4096 tokens for most models.

- **Example**:
  ```python
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": "Tell me a story."}],
      max_tokens=150
  )
  ```

**Function Calling**

- **Description**: This feature allows the model to invoke predefined functions based on user input, facilitating interaction with external APIs or services.

- **Example**:

  ```python
  functions = [
      {
          "name": "get_current_weather",
          "description": "Get the current weather for a location.",
          "parameters": {
              "type": "object",
              "properties": {
                  "location": {"type": "string"},
                  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
              },
              "required": ["location"]
          }
      }
  ]

  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": "What's the weather like in New York?"}],
      functions=functions,
      function_call={"name": "get_current_weather"}
  )
  ```

### Roles in API Calls

Understanding the roles involved in an API call helps structure interactions effectively.

**System Role**

- **Purpose**: Provides high-level instructions or context that guides the model's behavior throughout the conversation.
- **Usage**: Set at the beginning of the message array to establish tone or rules for interaction.

- **Example**:
  ```python
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What can you do?"}
  ]
  ```

**User Role**

- **Purpose**: Represents inputs from the human user, guiding the conversation with queries or prompts.
- **Usage**: Used most frequently in interactions to ask questions or provide statements.

- **Example**:
  ```python
  {"role": "user", "content": "Can you explain how OpenAI works?"}
  ```

**Assistant Role**

- **Purpose**: Represents responses generated by the model based on user inputs and system instructions.
- **Usage**: Automatically assumed by the model when replying to user queries.

- **Example**:
  ```python
  {"role": "assistant", "content": "OpenAI uses advanced machine learning techniques to generate text."}
  ```

### Additional Parameters

**Stream**

- **Description**: If set to true, allows streaming of partial responses as they are generated, useful for real-time applications.

**Logprobs**

- **Description**: Returns log probabilities of token predictions, useful for understanding model behavior and improving outputs.

### Conclusion

The OpenAI API offers a robust set of parameters and roles that enable developers to create highly interactive applications. By adjusting parameters like temperature and utilizing structured roles effectively, users can tailor responses to meet specific needs while ensuring clarity and control in conversations.

Here are examples of how to use the OpenAI API with various providers that are compatible, along with an example of a lightweight language model (LLM).

## OpenAI API Examples with Compatible Providers

### 1. **Groq**

To use Groq's API, you need to set the base URL and provide your Groq API key.

```python
import os
import openai

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

response = client.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Tell me a joke."}]
)
print(response.choices[0].message['content'])
```

### 2. **Mistral AI**

Mistral AI also provides an OpenAI-compatible API. Here's how to use it:

```python
import requests

url = "https://api.mistral.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ.get('MISTRAL_API_KEY')}",
    "Content-Type": "application/json"
}
data = {
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}

response = requests.post(url, headers=headers, json=data)
print(response.json()['choices'][0]['message']['content'])
```

### 3. **Hugging Face**

Using Hugging Face's API requires an access token. Here's an example:

```python
import requests

url = "https://api-inference.huggingface.co/models/gpt2"
headers = {
    "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"
}
data = {
    "inputs": "Once upon a time in a land far away,"
}

response = requests.post(url, headers=headers, json=data)
print(response.json()[0]['generated_text'])
```

### 4. **Google Vertex AI**

To interact with Google Vertex AI, you can use the following code:

```python
from google.cloud import aiplatform

aiplatform.init(project='your-project-id', location='us-central1')

response = aiplatform.gapic.PredictionServiceClient().predict(
    endpoint='projects/your-project-id/locations/us-central1/endpoints/your-endpoint-id',
    instances=[{"content": "Who won the World Series in 2020?"}],
)
print(response.predictions)
```

### 5. **Microsoft Azure OpenAI**

Here's how to call Azure's OpenAI service:

```python
import requests

url = f"https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name/chat/completions?api-version=2023-05-15"
headers = {
    "Content-Type": "application/json",
    "api-key": os.environ.get("AZURE_OPENAI_API_KEY")
}
data = {
    "messages": [{"role": "user", "content": "What's the weather today?"}],
    "model": "gpt-35-turbo"
}

response = requests.post(url, headers=headers, json=data)
print(response.json()['choices'][0]['message']['content'])
```

### 6. **Anthropic**

Using Anthropic's Claude model via its API can be done as follows:

```python
import requests

url = "https://api.anthropic.com/v1/complete"
headers = {
    "Authorization": f"Bearer {os.environ.get('ANTHROPIC_API_KEY')}",
    "Content-Type": "application/json"
}
data = {
    "model": "claude-v1",
    "prompt": "Explain quantum physics in simple terms.",
}

response = requests.post(url, headers=headers, json=data)
print(response.json()['completion'])
```

This example demonstrates how to utilize lightweight models effectively while still achieving meaningful outputs in text generation tasks.

Sources:

https://www.coltsteele.com/tips/understanding-openai-s-temperature-parameter
https://community.make.com/t/what-is-the-difference-between-system-user-and-assistant-roles-in-chatgpt/36160
https://arize.com/blog-course/mastering-openai-api-tips-and-tricks/
https://learn.microsoft.com/ko-kr/Azure/ai-services/openai/reference
https://community.openai.com/t/the-system-role-how-it-influences-the-chat-behavior/87353
https://community.openai.com/t/understanding-role-management-in-openais-api-two-methods-compared/253289
https://platform.openai.com/docs/advanced-usage
https://platform.openai.com/docs/api-reference

https://console.groq.com/docs/openai
https://docs.jabref.org/ai/ai-providers-and-api-keys
https://www.reddit.com/r/LocalLLaMA/comments/16csz5n/best_openai_api_compatible_application_server/
https://docs.gptscript.ai/alternative-model-providers
https://towardsdatascience.com/how-to-build-an-openai-compatible-api-87c8edea2f06?gi=5537ceb80847

https://modelfusion.dev/integration/model-provider/openaicompatible/
https://docs.jabref.org/ai/ai-providers-and-api-keys
https://docs.gptscript.ai/alternative-model-providers
https://console.groq.com/docs/openai
