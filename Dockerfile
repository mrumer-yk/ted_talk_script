# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create output directory
RUN mkdir -p /app/out

# Expose port (Railway will set the PORT environment variable)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Update web_app.py to use PORT environment variable
RUN sed -i 's/port=7864/port=int(os.environ.get("PORT", 8080))/' web_app.py

# Start the application
CMD ["python", "web_app.py"]
