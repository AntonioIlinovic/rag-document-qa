FROM python:3.12-slim

WORKDIR /app

# Copy pyproject.toml first for better caching
COPY pyproject.toml ./

# Install uv for faster dependency management
RUN pip install uv
RUN uv pip install --system -e .

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]