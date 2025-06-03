# Document Q&A AI Agent

An advanced document processing and question-answering system that combines local document analysis with Arxiv research paper integration.

> This project is an enhanced version of the [Multi-modal Document Q&A System](https://github.com/RushiChaganti/Multi_modal_Doc_QA_system) with added Arxiv research paper integration and improved document management capabilities.

## Features

### Document Processing
- Process multiple document types (PDF, TXT)
- Extract structured information:
  - Sections and subsections
  - Tables and figures
  - References
- OCR capabilities for text extraction from images
- Intelligent text chunking for better context understanding

### Arxiv Integration
- Search and download research papers directly from Arxiv
- Process papers for Q&A without manual downloading
- Maintain a collection of papers for easy reference
- Combine multiple papers for cross-document analysis

### Multi-Modal Capabilities
- Text extraction from documents
- Image processing with OCR
- Table detection and extraction
- Structured content analysis

### Advanced Q&A
- Chat with multiple documents simultaneously
- Context-aware responses
- Source tracking for answers
- Support for various LLM providers:
  - OpenAI
  - Anthropic
  - Google Gemini
  - Hugging Face
  - Ollama

## Getting Started

1. Install uv (if not already installed):
```bash
pip install uv
```

2. Create and activate a virtual environment using uv:
```bash
# Create a new virtual environment
uv venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

4. Set up your API keys in the application:
   - OpenAI API key (for OpenAI models)
   - Anthropic API key (for Claude models)
   - Google API key (for Gemini models)
   - Hugging Face API token (for Hugging Face models)

5. Run the application:
```bash
uv run streamlit run app.py
```

## Usage

### Working with Local Documents
1. Upload PDF or TXT files through the interface
2. The system will process and index the documents
3. Ask questions about the content
4. View structured information and extracted tables

### Working with Arxiv Papers
1. Search for papers using keywords
2. Browse search results with paper details
3. Add papers to your collection
4. Chat with multiple papers simultaneously
5. Remove papers from collection when no longer needed

### Document Collection Management
- View all documents in your collection
- See document metadata and statistics
- Remove documents as needed
- Chat with the entire collection at once

## Requirements

See `requirements.txt` for a complete list of dependencies.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

This project builds upon the [Multi-modal Document Q&A System](https://github.com/RushiChaganti/Multi_modal_Doc_QA_system) by RushiChaganti, adding enhanced features for research paper integration and improved document management.
