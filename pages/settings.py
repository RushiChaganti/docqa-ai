import streamlit as st
import os
from providers.openai_provider import get_openai_models, get_openai_llm, get_openai_embeddings
from providers.anthropic_provider import get_anthropic_models, get_anthropic_llm
from providers.gemini_provider import get_gemini_models, get_gemini_llm
from providers.huggingface_provider import get_huggingface_models, get_huggingface_llm, get_huggingface_embeddings
from providers.ollama_provider import get_ollama_models, get_ollama_llm

st.set_page_config(page_title="Settings - Document Q&A AI Agent", layout="wide")

st.title("Settings")
st.markdown("Configure your model and embedding settings here.")

# Model Provider Selection
st.header("Model Provider")
model_provider = st.selectbox(
    "Select Model Provider",
    ["OpenAI", "Anthropic", "Google Gemini", "Hugging Face", "Ollama"],
    index=["OpenAI", "Anthropic", "Google Gemini", "Hugging Face", "Ollama"].index(st.session_state.model_settings["provider"])
)

# API Keys and Model Configurations based on provider
if model_provider == "OpenAI":
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Fetch OpenAI models
    if st.button("Refresh OpenAI Models"):
        with st.spinner("Fetching OpenAI models..."):
            st.session_state.openai_models = get_openai_models()
    
    # Display available models or default options
    if st.session_state.openai_models:
        model_name = st.selectbox("Select OpenAI Model", st.session_state.openai_models, index=st.session_state.openai_models.index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in st.session_state.openai_models else 0)
    else:
        model_name = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4"], index=["gpt-3.5-turbo", "gpt-4"].index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in ["gpt-3.5-turbo", "gpt-4"] else 0)
        if api_key:
            st.info("Click 'Refresh OpenAI Models' to fetch available models.")
    
    embedding_provider = "OpenAI"
    
elif model_provider == "Anthropic":
    api_key = st.text_input("Enter your Anthropic API Key:", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""))
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Fetch Anthropic models
    if st.button("Refresh Anthropic Models"):
        with st.spinner("Fetching Anthropic models..."):
            st.session_state.anthropic_models = get_anthropic_models()
    
    # Display available models
    if st.session_state.anthropic_models:
        model_name = st.selectbox("Select Anthropic Model", st.session_state.anthropic_models, index=st.session_state.anthropic_models.index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in st.session_state.anthropic_models else 0)
    else:
        model_name = st.selectbox("Select Model", get_anthropic_models(), index=get_anthropic_models().index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in get_anthropic_models() else 0)
    
    embedding_provider = st.selectbox("Embedding Provider", ["OpenAI", "HuggingFace"], index=["OpenAI", "HuggingFace"].index(st.session_state.model_settings["embedding_provider"]))
    
elif model_provider == "Google Gemini":
    api_key = st.text_input("Enter your Google API Key:", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Fetch Gemini models
    if st.button("Refresh Gemini Models"):
        with st.spinner("Fetching Gemini models..."):
            st.session_state.gemini_models = get_gemini_models(api_key)
    
    # Display available models or default options
    if st.session_state.gemini_models:
        model_name = st.selectbox("Select Gemini Model", st.session_state.gemini_models, index=st.session_state.gemini_models.index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in st.session_state.gemini_models else 0)
    else:
        model_name = st.selectbox("Select Model", ["gemini-pro", "gemini-1.5-pro"], index=["gemini-pro", "gemini-1.5-pro"].index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in ["gemini-pro", "gemini-1.5-pro"] else 0)
        if api_key:
            st.info("Click 'Refresh Gemini Models' to fetch available models.")
    
    embedding_provider = st.selectbox("Embedding Provider", ["OpenAI", "HuggingFace"], index=["OpenAI", "HuggingFace"].index(st.session_state.model_settings["embedding_provider"]))
    
elif model_provider == "Hugging Face":
    api_key = st.text_input("Enter your Hugging Face API Key:", type="password", value=os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""))
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    # Fetch HuggingFace models
    if st.button("Refresh HuggingFace Models"):
        with st.spinner("Fetching HuggingFace models..."):
            st.session_state.huggingface_models = get_huggingface_models(api_key)
    
    # Display available models or manual input
    if st.session_state.huggingface_models:
        model_name = st.selectbox("Select HuggingFace Model", st.session_state.huggingface_models, index=st.session_state.huggingface_models.index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in st.session_state.huggingface_models else 0)
    else:
        model_name = st.text_input("Model Name (e.g., google/flan-t5-xxl)", value=st.session_state.model_settings["model_name"])
        if api_key:
            st.info("Click 'Refresh HuggingFace Models' to fetch popular models.")
    
    embedding_provider = "HuggingFace"
    
elif model_provider == "Ollama":
    ollama_base_url = st.text_input("Ollama Base URL", value=st.session_state.model_settings["ollama_base_url"])
    
    # Fetch Ollama models
    if st.button("Refresh Ollama Models"):
        with st.spinner("Fetching Ollama models..."):
            st.session_state.ollama_models = get_ollama_models(ollama_base_url)
    
    # Display available models or manual input
    if st.session_state.ollama_models:
        model_name = st.selectbox("Select Ollama Model", st.session_state.ollama_models, index=st.session_state.ollama_models.index(st.session_state.model_settings["model_name"]) if st.session_state.model_settings["model_name"] in st.session_state.ollama_models else 0)
    else:
        model_name = st.text_input("Model Name (e.g., llama2)", value=st.session_state.model_settings["model_name"])
        st.info("No Ollama models found. Make sure Ollama is running and click 'Refresh Ollama Models'.")
    
    embedding_provider = "HuggingFace"

# Embedding model selection for HuggingFace
if embedding_provider == "HuggingFace":
    hf_embedding_model = st.text_input(
        "HuggingFace Embedding Model", 
        value=st.session_state.model_settings["hf_embedding_model"]
    )

# Temperature setting
temperature = st.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.model_settings["temperature"], 
    step=0.1
)

# Save settings button
if st.button("Save Settings", type="primary"):
    st.session_state.model_settings.update({
        "provider": model_provider,
        "model_name": model_name,
        "temperature": temperature,
        "embedding_provider": embedding_provider,
        "hf_embedding_model": hf_embedding_model if embedding_provider == "HuggingFace" else st.session_state.model_settings["hf_embedding_model"],
        "ollama_base_url": ollama_base_url if model_provider == "Ollama" else st.session_state.model_settings["ollama_base_url"]
    })
    st.success("Settings saved successfully!")
    st.rerun()

# Back to main page button
if st.button("Back to Main Page"):
    st.switch_page("app.py") 