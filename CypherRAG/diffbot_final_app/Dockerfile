FROM python:3.11-slim

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Set the working directory
WORKDIR /code

# Copy necessary files for Poetry
COPY ./pyproject.toml ./README.md ./poetry.lock* ./

# Copy the .env file into the Docker image
COPY ./.env ./.env

# Install dependencies without root packages
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the application code
COPY ./app ./app

# Expose the application port
EXPOSE 8080

# Run the application
CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080
