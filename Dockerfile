# HippoChat/Dockerfile

FROM python:3.11-slim

WORKDIR /HippoChat



RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -

# ENV POETRY_HOME="/etc/poetry"
ENV PATH="/etc/poetry/bin:$PATH"

RUN git clone https://github.com/AnselmJeong/HippoChat.git .

RUN poetry install

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./hippochat/app.py", "--server.port=8501", "--server.address=0.0.0.0"]