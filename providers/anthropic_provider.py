import os
import requests
from langchain_anthropic import ChatAnthropic

def get_anthropic_models():
    """Fetch available Anthropic models"""
    # Anthropic doesn't have a public model listing API, so we use a predefined list
    return [
        # Claude 3 Models (Latest)
        "claude-3-opus-20240229",    # Most capable model
        "claude-3-sonnet-20240229",  # Balanced performance
        "claude-3-haiku-20240307",   # Fastest and most efficient
        
        # Claude 2 Models
        "claude-2.1",                # Previous generation
        "claude-2.0",                # Previous generation
        
        # Claude Instant Models
        "claude-instant-1.2"         # Fast, lightweight model
    ]

def get_anthropic_llm(model_name, temperature):
    """Initialize Anthropic LLM"""
    return ChatAnthropic(model=model_name, temperature=temperature)