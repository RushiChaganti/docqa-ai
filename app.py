import streamlit as st
import os
from langchain.chains import RetrievalQA
from typing import List, Dict, Any
from langchain.vectorstores import FAISS

# Import provider modules
from providers.openai_provider import get_openai_models, get_openai_llm, get_openai_embeddings
from providers.anthropic_provider import get_anthropic_models, get_anthropic_llm
from providers.gemini_provider import get_gemini_models, get_gemini_llm
from providers.huggingface_provider import get_huggingface_models, get_huggingface_llm, get_huggingface_embeddings
from providers.ollama_provider import get_ollama_models, get_ollama_llm
from tools.document_processor import EnhancedDocumentProcessor
from tools.arxiv_tools import ArxivTools

# Set page configuration
st.set_page_config(page_title="Document Q&A AI Agent", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "document_collections" not in st.session_state:
    st.session_state.document_collections = {}
if "openai_models" not in st.session_state:
    st.session_state.openai_models = []
if "anthropic_models" not in st.session_state:
    st.session_state.anthropic_models = []
if "gemini_models" not in st.session_state:
    st.session_state.gemini_models = []
if "huggingface_models" not in st.session_state:
    st.session_state.huggingface_models = []
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []
if "model_settings" not in st.session_state:
    st.session_state.model_settings = {
        "provider": "OpenAI",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.0,
        "embedding_provider": "OpenAI",
        "hf_embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "ollama_base_url": "http://localhost:11434"
    }

# App title and description
st.title("Document Q&A AI Agent")
st.markdown("""
This system allows you to:
- Upload and process multiple documents
- Extract structured information (sections, tables, references)
- Query content using natural language
- Search and download papers from Arxiv
""")

# Settings button in the top right
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("⚙️ Settings"):
        st.switch_page("pages/settings.py")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Local Document Upload
    st.subheader("Upload Local Documents")
    uploaded_files = st.file_uploader("Upload documents (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    
    # Show uploaded files and confirmation
    if uploaded_files:
        # Add confirmation button
        if st.button("Process Uploaded Files", type="primary"):
            if ((st.session_state.model_settings["provider"] in ["OpenAI", "Anthropic", "Google Gemini", "Hugging Face"] and 
                 os.environ.get(f"{st.session_state.model_settings['provider'].upper()}_API_KEY")) or 
                st.session_state.model_settings["provider"] == "Ollama"):
                
                with st.spinner("Processing documents..."):
                    # Create embeddings based on selected provider
                    if st.session_state.model_settings["embedding_provider"] == "OpenAI" and "OPENAI_API_KEY" in os.environ:
                        embeddings = get_openai_embeddings()
                    elif st.session_state.model_settings["embedding_provider"] == "HuggingFace":
                        embeddings = get_huggingface_embeddings(st.session_state.model_settings["hf_embedding_model"])
                    
                    # Process documents
                    processor = EnhancedDocumentProcessor()
                    processed_count = 0
                    failed_files = []
                    
                    for uploaded_file in uploaded_files:
                        try:
                            result = processor.process_document(uploaded_file, embeddings)
                            doc_id = f"local_{uploaded_file.name}"
                            st.session_state.document_collections[doc_id] = {
                                'type': 'local',
                                'name': uploaded_file.name,
                                'vectorstore': result['vectorstore'],
                                'metadata': result['metadata'],
                                'structured_content': result['structured_content'],
                                'tables': result['tables']
                            }
                            processed_count += 1
                        except Exception as e:
                            failed_files.append((uploaded_file.name, str(e)))
                    
                    # Show processing results
                    if processed_count > 0:
                        st.success(f"Successfully processed {processed_count} out of {len(uploaded_files)} documents")
                    
                    if failed_files:
                        st.error("Failed to process some files:")
                        for file_name, error in failed_files:
                            st.error(f"- {file_name}: {error}")
                    
                    # Update vectorstore with all processed documents
                    if processed_count > 0:
                        all_chunks = []
                        for collection in st.session_state.document_collections.values():
                            all_chunks.extend(collection['vectorstore'].docstore._dict.values())
                        
                        # Create combined vectorstore
                        st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)
                        st.rerun()
            else:
                st.error("Please configure your model settings in the Settings page (⚙️ button in the top right).")
    else:
        st.info("Upload PDF or TXT files to process them.")

    # Arxiv Search
    st.subheader("Search Arxiv Papers")
    arxiv_query = st.text_input("Search Arxiv papers")
    if arxiv_query:
        arxiv_tools = ArxivTools()
        with st.spinner("Searching Arxiv..."):
            papers = arxiv_tools.search_papers(arxiv_query)
            for paper in papers:
                with st.expander(f"{paper['title']} ({paper['published'].strftime('%Y-%m-%d')})"):
                    st.write("**Authors:**", ", ".join(paper['authors']))
                    st.write("**Abstract:**", paper['abstract'])
                    st.write("**Published Date:**", paper['published'].strftime('%Y-%m-%d'))
                    st.write("**PDF URL:**", paper['pdf_url'])
                    if st.button(f"Add to collection", key=paper['entry_id']):
                        with st.spinner("Processing paper..."):
                            # Create embeddings based on selected provider
                            if st.session_state.model_settings["embedding_provider"] == "OpenAI" and "OPENAI_API_KEY" in os.environ:
                                embeddings = get_openai_embeddings()
                            elif st.session_state.model_settings["embedding_provider"] == "HuggingFace":
                                embeddings = get_huggingface_embeddings(st.session_state.model_settings["hf_embedding_model"])
                            
                            # Process paper for chat
                            result = arxiv_tools.process_paper_for_chat(paper['pdf_url'], embeddings)
                            
                            # Add to document collections
                            paper_id = paper['entry_id']
                            st.session_state.document_collections[paper_id] = {
                                'type': 'arxiv',
                                'paper': paper,
                                'vectorstore': result['vectorstore'],
                                'metadata': result['metadata']
                            }
                            
                            st.success(f"Paper added to collection: {paper['title']}")
                            st.info(f"Number of pages: {result['metadata']['num_pages']}, Chunks: {result['metadata']['chunks']}")
    # List all documents
    st.sidebar.markdown("**Documents in Collection:**")
    for doc_id, collection in st.session_state.document_collections.items():
        if collection['type'] == 'arxiv':
            st.sidebar.write(collection['paper']['title'])
        else:
            st.sidebar.write(collection['name'])
    # Display collection info if available
    if st.session_state.document_collections:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Document Collection")
        st.sidebar.write(f"**Total Documents:** {len(st.session_state.document_collections)}")
        
    st.sidebar.markdown("---")

    # Document Collection Management
    st.subheader("Your Document Collection")
    if st.session_state.document_collections:
        # Display current collection
        for doc_id, collection in st.session_state.document_collections.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if collection['type'] == 'arxiv':
                    paper = collection['paper']
                    st.write(f"**{paper['title']}** (Arxiv Paper)")
                    st.write(f"Authors: {', '.join(paper['authors'])}")
                    st.write(f"Published: {paper['published'].strftime('%Y-%m-%d')}")
                else:
                    st.write(f"**{collection['name']}** (Local Document)")
                    if 'structured_content' in collection:
                        st.write(f"Sections: {len(collection['structured_content']['sections'])}")
                        st.write(f"Tables: {len(collection['tables'])}")
            with col2:
                if st.button("Remove", key=f"remove_{doc_id}"):
                    del st.session_state.document_collections[doc_id]
                    st.success(f"Removed document: {collection['paper']['title'] if collection['type'] == 'arxiv' else collection['name']}")
                    st.rerun()

        # Create combined vectorstore for all documents
        if st.session_state.document_collections:
            all_chunks = []
            for collection in st.session_state.document_collections.values():
                all_chunks.extend(collection['vectorstore'].docstore._dict.values())
            
            # Create embeddings based on selected provider
            if st.session_state.model_settings["embedding_provider"] == "OpenAI" and "OPENAI_API_KEY" in os.environ:
                embeddings = get_openai_embeddings()
            elif st.session_state.model_settings["embedding_provider"] == "HuggingFace":
                embeddings = get_huggingface_embeddings(st.session_state.model_settings["hf_embedding_model"])
            
            # Create combined vectorstore
            st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)
            st.success(f"Ready to chat with {len(st.session_state.document_collections)} documents!")
    else:
        st.info("No documents in collection. Upload local documents or add Arxiv papers above.")

# Main chat interface
if (st.session_state.model_settings["provider"] in ["OpenAI", "Anthropic", "Google Gemini", "Hugging Face"] and 
    (st.session_state.model_settings["provider"] == "OpenAI" and "OPENAI_API_KEY" in os.environ or
     st.session_state.model_settings["provider"] == "Anthropic" and "ANTHROPIC_API_KEY" in os.environ or
     st.session_state.model_settings["provider"] == "Google Gemini" and "GOOGLE_API_KEY" in os.environ or
     st.session_state.model_settings["provider"] == "Hugging Face" and "HUGGINGFACEHUB_API_TOKEN" in os.environ)) or st.session_state.model_settings["provider"] == "Ollama":
    
    
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for user question
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response if vectorstore exists
        if st.session_state.vectorstore:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Initialize the appropriate LLM based on selection
                    if st.session_state.model_settings["provider"] == "OpenAI":
                        llm = get_openai_llm(st.session_state.model_settings["model_name"], st.session_state.model_settings["temperature"])
                    elif st.session_state.model_settings["provider"] == "Anthropic":
                        llm = get_anthropic_llm(st.session_state.model_settings["model_name"], st.session_state.model_settings["temperature"])
                    elif st.session_state.model_settings["provider"] == "Google Gemini":
                        llm = get_gemini_llm(st.session_state.model_settings["model_name"], st.session_state.model_settings["temperature"])
                    elif st.session_state.model_settings["provider"] == "Hugging Face":
                        llm = get_huggingface_llm(st.session_state.model_settings["model_name"], st.session_state.model_settings["temperature"])
                    elif st.session_state.model_settings["provider"] == "Ollama":
                        llm = get_ollama_llm(st.session_state.model_settings["model_name"], st.session_state.model_settings["ollama_base_url"], st.session_state.model_settings["temperature"])
                    
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 5}  # Increased to get more context from all documents
                        ),
                        return_source_documents=True
                    )
                    
                    # Get response
                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"]
                    
                    # Display answer
                    st.write(answer)
                    
                    # Display sources
                    with st.expander("Sources"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.chat_message("assistant"):
                st.write("Please add some documents to the collection first.")
                st.session_state.messages.append({"role": "assistant", "content": "Please add some documents to the collection first."})
else:
    st.info(f"Please configure your model settings in the Settings page (⚙️ button in the top right).")
