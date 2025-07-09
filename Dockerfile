# Dockerfile for Streamlit Application
# We'll build this step by step

# Step 1: Choose base image
FROM python:3.11-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Step 4: Install Python dependencies
RUN pip install -r requirements.txt

# Step 5: Copy all application files
COPY . .

# Step 6: Expose port 8501 (Streamlit's default port)
EXPOSE 8501

# Step 7: Define the command to run when container starts
CMD ["streamlit", "run", "streamlit_reappro.py", "--server.port=8501", "--server.address=0.0.0.0"] 