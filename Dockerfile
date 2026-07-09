# Use official lightweight Python image
FROM python:3.9-slim

# Install system dependencies (needed for certain libraries if required, e.g., image reading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Create the uploads directory and set permissions
RUN mkdir -p uploads && chmod 777 uploads

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
