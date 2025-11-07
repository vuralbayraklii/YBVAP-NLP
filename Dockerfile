FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (gerekirse)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Port (uygulamanıza göre ayarlayın)
EXPOSE 8000

# Uygulamanızı başlatma komutu (örnek)
CMD ["python", "Model/test2.py"]