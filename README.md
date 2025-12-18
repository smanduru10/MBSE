# MBSE - Meta Model Completion

## 📁 Repository Structure
```
.
MBSE/
│
├── KnowledgeGraph/
│   │
│   ├── KGConstruction/
│   │   ├── knowledge_graph.pkl       # Serialized knowledge graph object
│   │   ├── poc_knowledge_graph.html  # PoC HTML graph (from a small sample of the modelset dataset in HTML Format)
│   │   
│   ├── kg_Construct.py               # End-to-end KG construction from modelset dataset 
│
└── README.md                         # Project overview, setup, and usage

```

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
