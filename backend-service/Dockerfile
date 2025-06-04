FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /queree
    
# Dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /queree

# Requirements
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI port 
EXPOSE 8000

# start Uvicorn server 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
