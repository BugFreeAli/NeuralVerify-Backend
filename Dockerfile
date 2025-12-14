# 1. Base Image (Official Python 3.10 Slim - Small & Fast)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (Required for some image libraries)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements first (Caching strategy to speed up builds)
COPY requirements.txt .

# 5. Install Dependencies (No Cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the app (Code + Model)
COPY main.py .
COPY with_flux_model.keras .

# 7. Expose the port (Render uses 10000 by default usually, but we set it below)
EXPOSE 8000

# 8. Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]