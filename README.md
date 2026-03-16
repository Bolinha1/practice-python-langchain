# Practice Python LangChain

A collection of Python scripts for practicing LangChain concepts, including chains, pipelines, and summarization.

## Requirements

- Python 3.10+

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd practice-python-langchain

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

## Running

Each script can be run individually:

```bash
# Fundamentals
python 1-fundamentos/1-hello-world.py
python 1-fundamentos/2-init-chat-model.py
python 1-fundamentos/3-prompt-template.py
python 1-fundamentos/4-chat-prompt-template.py

# Chain process
python 2-chain-process/1-iniciando-com-chain.py
python 2-chain-process/2-chain-com-decorator.py
python 2-chain-process/3-runnable-lambda.py
python 2-chain-process/4-pipeline-processamento.py
python 2-chain-process/5-sumarizacao.py
python 2-chain-process/6-sumarizacao-map-reduce.py
python 2-chain-process/7-pipeline-sumarizacao.py
```
