import requests
from langchain_community.llms import Ollama

def get_ollama_models(base_url="http://localhost:11434"):
    """Fetch available Ollama models"""
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            # Extract model names from the response
            model_names = [model['name'] for model in models_data.get('models', [])]
            return model_names
        else:
            return []
    except:
        return []

def get_ollama_llm(model_name, base_url, temperature):
    """Initialize Ollama LLM"""
    return Ollama(model=model_name, base_url=base_url, temperature=temperature)