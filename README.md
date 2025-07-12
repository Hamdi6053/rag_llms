PDF Document Analysis and Data Extraction (with RAG)

This project is a web application that uses the Retrieval-Augmented Generation (RAG) technique to extract structured data from user-uploaded PDF documents.
The application leverages the deepseek-coder language model, running locally via Ollama, to extract metadata such as the author, title, and year in JSON format.
Pydantic ensures the structure and validity of this data, and the results are presented to the user as a table (DataFrame) in a Streamlit interface. 
The entire project is containerized with Docker and Docker Compose, making setup and deployment straightforward.

Features

PDF Upload: User-friendly interface to upload PDF files.
RAG-based Data Extraction: Intelligent data extraction by providing document content as context to a local LLM (Ollama & deepseek-coder).
Structured Output: Validates and structures the data from the LLM into a JSON format using Pydantic models.
Dynamic Table Display: Converts the extracted data into a Pandas DataFrame and displays it as a table in Streamlit.
Containerization: Easy setup and portability with Docker and Docker Compose.

Technology Stack

Backend: Python
Web Framework: Streamlit
LLM & RAG: Ollama, deepseek-coder
Data Validation: Pydantic
Containerization: Docker, Docker Compose
Libraries: PyMuPDF, Pandas, LangChain

