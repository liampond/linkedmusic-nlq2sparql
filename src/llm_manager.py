import os
import openai
import anthropic
# import google.generativeai as genai # Deprecated
from google import genai 
from typing import Optional

class LLMManager:
    def __init__(self, config):
        self.config = config
        self.setup_clients()

    def setup_clients(self):
        # We initialize clients on demand or upfront based on available keys
        pass

    def get_response(self, model_key: str, prompt: str, system_prompt: str = "") -> str:
        """
        Dispatches the request to the appropriate provider based on the configuration for 'model_key'.
        """
        model_config = self.config['models'].get(model_key)
        if not model_config:
            raise ValueError(f"Model configuration for '{model_key}' not found.")

        provider = model_config.get('provider')
        model_name = model_config.get('model_name')
        env_key = model_config.get('env_key')
        api_key = os.getenv(env_key)

        if not api_key:
            # For local testing or if keys are missing, we might want to return mock data
            # But for now, let's raise error
            raise ValueError(f"API Key for {model_key} ({env_key}) not found in environment variables.")

        if provider == "openai":
            return self._call_openai(api_key, model_name, prompt, system_prompt)
        elif provider == "anthropic":
            return self._call_anthropic(api_key, model_name, prompt, system_prompt)
        elif provider == "google":
            return self._call_google(api_key, model_name, prompt, system_prompt)
        elif provider == "openai_compatible":
            base_url = model_config.get('base_url')
            return self._call_openai(api_key, model_name, prompt, system_prompt, base_url=base_url)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _call_openai(self, api_key, model, prompt, system_prompt, base_url=None):
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI/Compatible: {e}"

    def _call_anthropic(self, api_key, model, prompt, system_prompt):
        client = anthropic.Anthropic(api_key=api_key)
        
        try:
            # Anthropic handles system prompts as a separate parameter in newer APIs
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error calling Anthropic: {e}"

    def _call_google(self, api_key, model, prompt, system_prompt):
        # Using new google.genai library
        try:
            client = genai.Client(api_key=api_key)
            
            # Simple content generation
            # Note: system instruction support in v1beta varies by signature
            # We construct a combined prompt if system param isn't directly exposed in generate_content 
            # effectively, or pass config.
            
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                   'system_instruction': system_prompt,
                   'temperature': 0 
                }
            )
            return response.text
        except Exception as e:
            return f"Error calling Google: {e}"
