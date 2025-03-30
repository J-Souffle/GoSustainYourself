FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project code
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose the port that Google Cloud will route to.
EXPOSE 8080

# Use Gunicorn to serve the app.
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8080"]