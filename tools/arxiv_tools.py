import arxiv
import tempfile
import os
from typing import List, Dict, Any
import requests
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

class ArxivTools:
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for papers on Arxiv based on query"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in self.client.results(search):
            results.append({
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.summary,
                'pdf_url': paper.pdf_url,
                'published': paper.published,
                'entry_id': paper.entry_id
            })
        
        return results
    
    def process_paper_for_chat(self, pdf_url: str, embeddings) -> Dict[str, Any]:
        """Download and process paper directly for chat"""
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download paper from {pdf_url}")
            
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix='.pdf')
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(response.content)
            
            # Load and process the PDF
            loader = PyPDFLoader(path)
            documents = loader.load()
            
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
                'metadata': {
                    'num_pages': len(documents),
                    'chunks': len(chunks)
                }
            }
        finally:
            # Clean up temporary file
            os.unlink(path)
    
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific paper"""
        search = arxiv.Search(id_list=[paper_id])
        paper = next(self.client.results(search))
        
        return {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'abstract': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': paper.published,
            'categories': paper.categories,
            'comment': paper.comment,
            'journal_ref': paper.journal_ref,
            'doi': paper.doi
        }
    
    def get_related_papers(self, paper_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get papers related to a specific paper"""
        paper_details = self.get_paper_details(paper_id)
        # Use title and abstract to find related papers
        query = f"ti:{paper_details['title']} OR abs:{paper_details['abstract'][:200]}"
        return self.search_papers(query, max_results) 