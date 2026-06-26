# Step 1: Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies needed for building packages if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy over requirements first to maximize Docker caching efficiency
COPY requirements.txt .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your local application code into the container
COPY . .

# Step 7: Expose ports for both FastAPI (8000) and MLflow UI (5000)
EXPOSE 8000
EXPOSE 5000

# Step 8: By default, we will boot up the FastAPI Server. 
# (We will use docker-compose to handle running both FastAPI and MLflow simultaneously!)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
