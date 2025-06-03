import os
import requests
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_huggingface_models(api_key):
    """Fetch popular HuggingFace models"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=20&filter=text-generation",
            headers=headers
        )
        
        if response.status_code == 200:
            models_data = response.json()
            model_ids = [model["id"] for model in models_data]
            return model_ids
        else:
            return ["google/flan-t5-xxl", "tiiuae/falcon-7b", "mistralai/Mistral-7B-v0.1"]
    except:
        return ["google/flan-t5-xxl", "tiiuae/falcon-7b", "mistralai/Mistral-7B-v0.1"]

def get_huggingface_llm(model_name, temperature):
    """Initialize HuggingFace LLM"""
    return HuggingFaceHub(
        repo_id=model_name,
        task="text-generation",
        model_kwargs={"temperature": temperature, "max_length": 512}
    )

def get_huggingface_embeddings(model_name):
    """Initialize HuggingFace embeddings"""
    return HuggingFaceEmbeddings(model_name=model_name)