# 1. Temel İmajı Belirleme
FROM python:3.11-slim 

# 2. Çalışma Dizini Oluşturma
WORKDIR /app

# 3. Sadece gereksinimler dosyasını kopyala ve kur
# Bu, kodunuz değişmediği sürece bu katmanın cache'lenmesini sağlar, build hızlanır.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Uygulama kodunu kopyala
# Yereldeki 'app' klasörünün İÇERİĞİNİ, container'daki '/app' dizinine kopyala.
COPY ./app .

# 5. Portu Dışarıya Açma
EXPOSE 8501

# 6. Konteyner Başlatıldığında Çalışacak Komut
# ENTRYPOINT yerine CMD kullanmak genellikle daha esnektir.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]