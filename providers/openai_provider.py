import os
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_openai_models():
    """Fetch available OpenAI models"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ["gpt-3.5-turbo", "gpt-4"]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        
        if response.status_code == 200:
            models_data = response.json()
            # Filter for chat models
            chat_models = [model["id"] for model in models_data["data"] 
                          if "gpt" in model["id"] and not model["id"].startswith("gpt-4-vision")]
            # Sort models
            chat_models.sort()
            return chat_models if chat_models else ["gpt-3.5-turbo", "gpt-4"]
        else:
            return ["gpt-3.5-turbo", "gpt-4"]
    except:
        return ["gpt-3.5-turbo", "gpt-4"]

def get_openai_llm(model_name, temperature):
    """Initialize OpenAI LLM"""
    return ChatOpenAI(model_name=model_name, temperature=temperature)

def get_openai_embeddings():
    """Initialize OpenAI embeddings"""
    return OpenAIEmbeddings()