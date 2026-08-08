# PyTorch wheels bundle the CUDA runtime, so no nvidia base image is needed.
# GPU access is provided by the NVIDIA Container Toolkit on the host.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Place the virtualenv inside the project so it is easy to inspect/cache
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

RUN pip install --no-cache-dir poetry

# Copy only the dependency manifest first for better layer caching.
# poetry.lock is optional on first build — Poetry will create it.
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY . .

EXPOSE 8765

CMD ["poetry", "run", "python", "-m", "solara", "run", "application/app.py", \
     "--host", "0.0.0.0", "--port", "8765", "--no-open"]
