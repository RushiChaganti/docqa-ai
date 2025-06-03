import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

def get_gemini_models(api_key):
    """Fetch available Gemini models"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Filter for text models that contain "gemini" in the name
        gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
        return gemini_models
    except:
        # Return default models if API call fails
        return ["gemini-pro", "gemini-1.5-pro"]

def get_gemini_llm(model_name, temperature):
    """Initialize Gemini LLM"""
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)