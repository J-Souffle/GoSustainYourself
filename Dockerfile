FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libmariadb-dev \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose the port for Google Cloud (Cloud Run listens on port 8080)
EXPOSE 8080

# Start the app with Gunicorn
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8080"]