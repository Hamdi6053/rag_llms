import streamlit as st  
from functions import *
import base64

def display_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_streamlit_page():
    st.set_page_config(layout="wide", page_title="LLM Tool")
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type="pdf")

    return col1, col2, uploaded_file

# Ana sayfayı oluştur
col1, col2, uploaded_file = load_streamlit_page()

# Dosya yüklendiyse işle
if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)

    # PDF'ten metin al
    documents = get_pdf_text(uploaded_file)

    # Vektör veritabanı oluştur (Artık API key yok, Ollama yerel çalışır)
    st.session_state.vector_store = create_vectorstore_from_texts(documents, uploaded_file.name)

    st.write("Input Processed")

# Cevap üretme
with col1:
    if st.button("Generate table"):
        with st.spinner("Generating answer"):
            # Vector store'dan yanıt al
            answer = query_document(
                vectorstore=st.session_state.vector_store, 
                query="Give me the title, summary, publication date, and authors of the research paper."
            )
            st.write(answer)
