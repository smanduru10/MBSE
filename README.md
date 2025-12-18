# MBSE - Meta Model Completion

## Environment Setup

Create a virtual environment to isolate dependencies:

```bash
python -m virtualenv mde-env
source mde-env/bin/activate
```

## Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Knowledge Graph Construction

Run the script that constructs the knowledge graph - (nodes: 12426 & edges: 73231)

Dataset: https://huggingface.co/datasets/antolin/modelset

```bash
python KnowledgeGraph/kg_construct.py
```
