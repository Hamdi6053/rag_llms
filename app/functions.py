import os
import re
import uuid
import hashlib
import tempfile
import pandas as pd
import logging

# Langchain Kütüphaneleri
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ollama_base_url():
    """
    Docker container içinde mi yoksa host'ta mı çalıştığını kontrol eder
    ve uygun Ollama URL'sini döner
    """
    # Environment variable'dan al (Docker Compose için)
    if 'OLLAMA_BASE_URL' in os.environ:
        return os.environ['OLLAMA_BASE_URL']
    
    # Docker container içinde mi kontrol et
    if os.path.exists('/.dockerenv'):
        return "http://host.docker.internal:11434"
    
    # Host makine
    return "http://localhost:11434"

def clean_filename(filename):
    """
    ChromaDB koleksiyon adı için dosya adını temizler
    """
    name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    if len(name) > 50:
        hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
        name = f"doc_{hash_suffix}"
    elif len(name) < 3:
        name = f"doc_{name}_collection"
    
    if not name or not name[0].isalnum():
        name = f"doc_{name}"
    if not name[-1].isalnum():
        name = f"{name}_doc"
    
    if len(name) < 3:
        name = "default_collection"
    
    return name[:63] # Uzunluk 63 karakteri geçmemeli

def get_pdf_text(uploaded_file): 
    """PDF dosyasını yükler ve document'leri döner"""
    try:
        input_file = uploaded_file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        
        logger.info(f"PDF loaded successfully with {len(documents)} pages")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise
    finally:
        if 'temp_file' in locals():
            os.unlink(temp_file.name)

def split_document(documents, chunk_size=1000, chunk_overlap=200):
    """Dokümanları chunk'lara böler"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Document split into {len(chunks)} chunks")
    return chunks

def get_embedding_function():
    """
    Ollama embedding fonksiyonunu başlatır.
    """
    try:
        base_url = get_ollama_base_url()
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url
        )
        logger.info(f"Embedding function initialized with base_url: {base_url}")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embedding function: {str(e)}")
        raise

def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    """
    Chunks'lardan vectorstore oluşturur
    """
    try:
        # Boş chunks kontrolü
        if not chunks:
            raise ValueError("No chunks provided for vectorstore creation")
        
        # Benzersiz chunk'ları filtrele
        unique_chunks = []
        seen_content = set()
        
        for doc in chunks:
            # Boş content kontrolü
            if not doc.page_content.strip():
                continue
                
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                unique_chunks.append(doc)
                seen_content.add(content_hash)
        
        if not unique_chunks:
            raise ValueError("No valid chunks found after filtering")
        
        logger.info(f"Creating vectorstore with {len(unique_chunks)} unique chunks")
        
        # Collection name temizle
        collection_name = clean_filename(file_name)
        
        # Eğer collection zaten varsa ve embedding boyutu uyumsuzsa, collection'ı sil
        try:
            # Mevcut collection'ı kontrol et
            import chromadb
            chroma_client = chromadb.PersistentClient(path=vector_store_path)
            
            # Collection'ın var olup olmadığını kontrol et
            existing_collections = chroma_client.list_collections()
            collection_exists = any(col.name == collection_name for col in existing_collections)
            
            if collection_exists:
                logger.info(f"Collection {collection_name} already exists. Deleting to avoid dimension mismatch.")
                chroma_client.delete_collection(collection_name)
                logger.info(f"Collection {collection_name} deleted successfully.")
                
        except Exception as e:
            logger.warning(f"Could not check/delete existing collection: {str(e)}")
        
        # Vectorstore oluştur - ID'leri otomatik generate et
        vectorstore = Chroma.from_documents(
            documents=unique_chunks,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=vector_store_path
        )
        
        logger.info(f"Vectorstore created successfully with collection: {collection_name}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {str(e)}")
        raise

def create_vectorstore_from_texts(documents, file_name):
    """
    Dokümanlardan vectorstore oluşturur
    """
    try:
        # Dokümanları chunk'lara böl
        docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
        
        # Embedding fonksiyonu al
        embedding_function = get_embedding_function()
        
        # Vectorstore oluştur
        vectorstore = create_vectorstore(docs, embedding_function, file_name)
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error in create_vectorstore_from_texts: {str(e)}")
        raise

def load_vectorstore(file_name, vectorstore_path="db"):
    """
    Mevcut vectorstore'u yükler
    """
    try:
        embedding_function = get_embedding_function()
        collection_name = clean_filename(file_name)
        
        vectorstore = Chroma(
            persist_directory=vectorstore_path, 
            embedding_function=embedding_function, 
            collection_name=collection_name
        )
        
        logger.info(f"Vectorstore loaded successfully: {collection_name}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        raise

class AnswerWithSources(BaseModel):
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")

class ExtractedInfoWithSources(BaseModel):
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources

def format_docs(docs):
    """Dokümanları formatlar"""
    return "\n\n".join(doc.page_content for doc in docs)

def query_document(vectorstore, query):
    """
    Vectorstore'dan sorgu yapar
    """
    try:
        # Docker için base_url parametresi eklendi
        base_url = get_ollama_base_url()
        llm = ChatOllama(
            model="deepseek-r1", 
            format="json", 
            temperature=0,
            base_url=base_url
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # En relevant 5 chunk'ı al
        )
        
        # Parser, LLM'den gelen JSON metnini Pydantic modeline dönüştürecek.
        parser = JsonOutputParser(pydantic_object=ExtractedInfoWithSources)

        # Prompt, LLM'e ne yapması gerektiğini ve çıktıyı nasıl formatlaması gerektiğini açıkça anlatıyor.
        prompt_template = ChatPromptTemplate.from_template(
            """You are an expert extraction assistant.
            From the context below, extract the information requested by the user.
            Format your answer as a JSON object that strictly follows the schema provided.

            Schema:
            {format_instructions}

            Context:
            {context}

            Question: {question}
            
            Important: 
            - Provide specific answers based on the context
            - Include relevant source text in the sources field
            - Explain your reasoning clearly
            - If information is not available in the context, state "Not available in the provided context"
            """
        )
        
        # RAG zinciri
        rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(),
                "format_instructions": lambda x: parser.get_format_instructions()
            }
            | prompt_template
            | llm
            | parser
        )

        # Zincir çalıştırılır ve sonuç bir Python sözlüğü olarak alınır.
        structured_response_dict = rag_chain.invoke(query)
        
        # Gelen sözlük DataFrame'e dönüştürülür.
        df = pd.DataFrame([structured_response_dict])

        answer_row = []
        source_row = []
        reasoning_row = []

        for col in df.columns:
            # Gelen sözlük yapısından verileri al.
            item = df[col][0]
            answer_row.append(item.get('answer', 'N/A'))
            source_row.append(item.get('sources', 'N/A'))
            reasoning_row.append(item.get('reasoning', 'N/A'))

        structured_response_df = pd.DataFrame(
            [answer_row, source_row, reasoning_row],
            columns=df.columns,
            index=['answer', 'source', 'reasoning']
        )

        return structured_response_df.T
        
    except Exception as e:
        logger.error(f"Error in query_document: {str(e)}")
        raise