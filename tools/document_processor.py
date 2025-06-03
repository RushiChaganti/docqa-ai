import tempfile
import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import pytesseract
from pdf2image import convert_from_path
import nltk
from nltk.tokenize import sent_tokenize
import re
from tabulate import tabulate

# Download required NLTK data
nltk.download('punkt')

class EnhancedDocumentProcessor:
    def __init__(self):
        pass
        
    def extract_structured_content(self, text: str) -> Dict[str, Any]:
        """Extract structured content from text including sections and references."""
        # Extract sections using simple pattern matching
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        for sent in sentences:
            # Check if sentence looks like a section header
            if re.match(r'^[A-Z][A-Za-z\s]+:', sent):
                if current_section:
                    sections[current_section] = ' '.join(current_content)
                current_section = sent.strip(':')
                current_content = []
            else:
                current_content.append(sent)
        
        if current_section:
            sections[current_section] = ' '.join(current_content)
            
        # Extract references
        references = []
        ref_pattern = r'\[\d+\].*?(?=\[\d+\]|$)'
        ref_matches = re.finditer(ref_pattern, text)
        for match in ref_matches:
            references.append(match.group().strip())
            
        return {
            'sections': sections,
            'references': references
        }
    
    def extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """Extract tables from text using pattern matching."""
        tables = []
        # Look for table-like structures
        table_pattern = r'(?:\|\s*[^\n]+\s*\|(?:\n\|\s*[^\n]+\s*\|)+)'
        table_matches = re.finditer(table_pattern, text)
        
        for match in table_matches:
            table_text = match.group()
            # Convert to DataFrame
            rows = [row.strip('|').split('|') for row in table_text.split('\n')]
            if len(rows) > 1:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                tables.append(df)
                
        return tables
    
    def process_document(self, uploaded_file, embeddings) -> Dict[str, Any]:
        """Process uploaded document and create vectorstore with enhanced features"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Load document based on file type
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                # Extract text from images in PDF
                images = convert_from_path(file_path)
                image_text = ""
                for image in images:
                    image_text += pytesseract.image_to_string(image) + "\n"
            else:
                loader = TextLoader(file_path)
                image_text = ""
                
            documents = loader.load()
            
            # Combine text from PDF and images
            full_text = "\n".join([doc.page_content for doc in documents]) + "\n" + image_text
            
            # Extract structured content
            structured_content = self.extract_structured_content(full_text)
            
            # Extract tables
            tables = self.extract_tables_from_text(full_text)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vectorstore
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            return {
                'vectorstore': vectorstore,
                'structured_content': structured_content,
                'tables': tables,
                'metadata': {
                    'filename': uploaded_file.name,
                    'num_pages': len(documents),
                    'num_tables': len(tables)
                }
            }
            
        finally:
            # Clean up temporary file
            os.unlink(file_path)
    
    def process_multiple_documents(self, uploaded_files: List[Any], embeddings) -> Dict[str, Any]:
        """Process multiple documents and combine their vectorstores"""
        processed_docs = []
        combined_chunks = []
        
        for file in uploaded_files:
            result = self.process_document(file, embeddings)
            processed_docs.append(result)
            combined_chunks.extend(result['vectorstore'].docstore._dict.values())
        
        # Create combined vectorstore
        combined_vectorstore = FAISS.from_documents(combined_chunks, embeddings)
        
        return {
            'vectorstore': combined_vectorstore,
            'processed_docs': processed_docs
        }

def process_document(uploaded_file, embeddings):
    """Legacy function for backward compatibility"""
    processor = EnhancedDocumentProcessor()
    result = processor.process_document(uploaded_file, embeddings)
    return result['vectorstore']