import json
import networkx as nx
from datasets import load_dataset
import pandas as pd
import pickle
from tqdm import tqdm
from collections import Counter
from pyvis.network import Network
import os

# Define relationship mapping
relationship_mapping = {
    ("EPackage", "EClass"): "contains_class",
    ("EPackage", "EEnum"): "contains_enum",
    ("EPackage", "EDataType"): "contains_datatype",
    ("EPackage", "EReference"): "contains_reference",

    ("EClass", "EAttribute"): "has_attribute",
    ("EClass", "EReference"): "has_reference",
    ("EClass", "EOperation"): "has_operation",
    ("EClass", "EClass"): "has_hierarchy",
    ("EClass", "EGenericType"): "uses_generic_type",
    ("EClass", "EPackage"): "is_contained_in",
    ("EClass", "ETypeParameter"): "declares_type_parameter",

    ("EReference", "EClass"): "references_class",
    ("EReference", "EGenericType"): "has_generic_reference",

    ("EAttribute", "EDataType"): "has_datatype",
    ("EAttribute", "EGenericType"): "has_generic_type",
    ("EAttribute", "EClass"): "belongs_to_class",
    ("EAttribute", "EEnum"): "has_enum_type",

    ("EEnum", "EEnumLiteral"): "has_literal",
    ("EEnum", "EDataType"): "is_enum_type_of",
    ("EEnum", "EPackage"): "belongs_to_package",

    ("EGenericType", "EClass"): "is_generic_version_of",
    ("EGenericType", "EDataType"): "is_generic_version_of",
    ("EGenericType", "ETypeParameter"): "has_type_parameter",
    ("EGenericType", "EEnum"): "uses_enum_type",

    ("EOperation", "EClass"): "belongs_to_class",
    ("EOperation", "EGenericType"): "returns_generic_type",

    ("EEnumLiteral", "EEnum"): "belongs_to_enum",
}

def load_existing_graph(path="knowledge_graph.pkl"):
    # print(path)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return nx.MultiDiGraph()

def save_graph(G, path="knowledge_graph.pkl"):
    with open(path, "wb") as f:
        pickle.dump(G, f)

def select_top_labels_balanced(df, label_col="labels", top_k=10, per_label=50, seed=42):
    top_labels = df[label_col].value_counts().nlargest(top_k).index.tolist()

    balanced_samples = []
    for label in top_labels:
        label_group = df[df[label_col] == label]
        sampled = label_group.sample(n=min(per_label, len(label_group)), random_state=seed)
        balanced_samples.append(sampled)

    return pd.concat(balanced_samples).reset_index(drop=True)

def visualize_knowledge_graph(KG, output_file="knowledge_graph.html"):
    """
    Creates an interactive visualization of the knowledge graph using Pyvis.
    :param KG: The MultiDiGraph representing the metamodel knowledge graph.
    :param output_file: The output HTML file to store the visualization.
    """
    print("nodes:", KG.number_of_nodes(), "edges:", KG.number_of_edges())
    if KG.number_of_nodes() == 0:
        print("KG is empty — nothing to visualize.")
    
    net = Network(notebook=False, 
                  directed=True, 
                  width="100%", 
                  height="800px",
                  cdn_resources="in_line"
                  )
    net.barnes_hut()
    
    # Add nodes with labels
    for node, data in KG.nodes(data=True):
        node_id = str(node)
        node_label = f"{data['name']} ({data['type']})"
        net.add_node(node_id, label=node_label, title=node_label, shape="ellipse", color="lightblue")

    # Add edges with relationship labels
    for u, v, data in KG.edges(data=True):
        edge_label = data.get("type", "related_to")
        net.add_edge(str(u), str(v), label=edge_label, title=edge_label, color="gray")

    net.show_buttons(filter_=['physics'])
    # net.show(output_file, notebook = False)
    net.write_html(output_file, notebook=False, open_browser=False)
    # print(f"Wrote: {output_file}")
    print(f"Interactive knowledge graph saved as {output_file}")


def process_batch(df_batch, G, node_mapping, start_id=0):
    
    node_id = start_id
    for _, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc="Processing batch"):
        node_info = {}
        graph_data = row['graph']
        label = row["labels"]

        for node in graph_data["nodes"]:
            node_name = node.get("name", "No Name").lower()
            node_type = node.get("eClass", "Unknown")
            node_key = (node_name, node_type)

            if not any(n[1]['name'] == node_name and n[1]['type'] == node_type for n in G.nodes(data=True)):
                G.add_node(node_id, name=node_name, type=node_type)
                node_mapping[node_id] = node_key
                node_id += 1

            node_info[node["id"]] = node_key

        for link in graph_data["links"]:
            source, target = link["source"], link["target"]
            source_type, target_type = node_info[source][1], node_info[target][1]
            relation = relationship_mapping.get((source_type, target_type), "related_to")
            
            # Get source and target node IDs based on name and type
            source_node_id = next((n for n, data in G.nodes(data=True)
                                   if data['name'] == node_info[source][0] and data['type'] == node_info[source][1]), None)
            target_node_id = next((n for n, data in G.nodes(data=True)
                                   if data['name'] == node_info[target][0] and data['type'] == node_info[target][1]), None)
            
            # Check if the exact same edge already exists using node IDs
            # Add edge if no duplicate exists using node IDs
            if source_node_id is not None and target_node_id is not None and not any(
                G[u][v][k]["type"] == relation for u, v, k in G.edges(keys=True)
                if u == source_node_id and v == target_node_id):
                
                G.add_edge(source_node_id, target_node_id, type=relation)

    return G, node_mapping, node_id


def build_knowledge_graph(df, batch_size=10, checkpoint_file="graph_checkpoint.txt"):
    
    G = load_existing_graph()
    # Determine which batch to start from based on checkpoint
    # print(checkpoint_file, os.path.exists(checkpoint_file))
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_batch = int(f.read().strip())
        f.close()
    else:
        start_batch = 0

    for i in range(start_batch * batch_size, len(df), batch_size):
        batch_idx = i // batch_size
        print(f"\n⏳ Processing batch {batch_idx} (rows {i} to {i + batch_size - 1})")

        G = load_existing_graph()
        node_mapping = {n: (data["name"], data["type"]) for n, data in G.nodes(data=True)}
        node_id = max(node_mapping.keys(), default=-1) + 1

        df_batch = df.iloc[i:i+batch_size]
        G, node_mapping, node_id = process_batch(df_batch, G, node_mapping, start_id=node_id)
        save_graph(G)

        # Update checkpoint file
        with open(checkpoint_file, "w") as f:
            f.write(str(batch_idx + 1))
        f.close()

    print("Knowledge graph build complete with checkpointing.")
    return G



def main():

    # Load and preprocess dataset
    ds = load_dataset("antolin/modelset", split="train").filter(lambda x: x['model_type'] == "ecore")
    ds = ds.filter(lambda x: not x['is_duplicated'])
    df = ds.to_pandas()
    df['graph'] = df['graph'].map(json.loads)
    df = df[['ids', 'graph', 'labels']]

    # balanced_df = select_top_labels_balanced(df, top_k=5, per_label=10)
    # balanced_df_path = "balanced_meta_models.csv"
    # balanced_df.to_csv(balanced_df_path, index=False)
    # print(balanced_df['labels'].value_counts())

    # Build KG
    G_final = build_knowledge_graph(df)

    # Print statistics
    print(f"Total nodes: {G_final.number_of_nodes()}, Total edges: {G_final.number_of_edges()}")
    edge_types = [data['type'] for _, _, data in G_final.edges(data=True)]
    for edge_type, count in Counter(edge_types).items():
        print(f"Edge type '{edge_type}': {count}")

    # Visualize KG
    visualize_knowledge_graph(G_final, "knowledge_graph.html")


if __name__ == "__main__":
    main()