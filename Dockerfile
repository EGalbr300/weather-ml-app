FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN pip install Flask gunicorn

# Copy the rest of the application code
COPY . .

# Run the app
CMD ["python3", "app.py"]
