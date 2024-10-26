FROM vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest

# Install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set default command with arguments
#CMD ["python", "/app/src/main.py", "--dataset", "CIFAR-10", "--epochs", "100"]