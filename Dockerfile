FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Version Label
LABEL version="v1"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the app
CMD ["python3", "app.py"]
