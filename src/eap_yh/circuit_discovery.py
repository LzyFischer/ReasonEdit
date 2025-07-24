import pdb
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from src.eap_yh.edge_attribute_patcher import EdgeAttributePatcher
import networkx as nx
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def find_node_importance(
    tokenizer, model, data: list[dict[str, str]], device: str
) -> list[tuple[str, float, torch.Tensor]]:
    """
    Find computation nodes that form a circuit based on causal effects.

    Returns:
        list[str]: Names of computation nodes with significant causal effects
    """
    patcher = EdgeAttributePatcher(tokenizer, model, device)
    effects = {n: [] for n in patcher.get_all_comp_nodes()}
    activations = {node: [] for node in effects}
    print(f"Processing {len(data)} input pairs...")
    count = 0
    for d in tqdm(data):
        # try:
        clean_outputs = patcher.get_activation(d["clean_text"], d["corrupted_text"])
        patcher.get_gradient(clean_outputs, d["answer"])
        cur_effects, cur_activations = {}, {}
        for node in patcher.get_all_comp_nodes():
            effect, activation = patcher.get_causal_effect(node)
            if effect.isnan().any() or effect.isinf().any():
                print("skipped")
                break
            cur_effects[node] = effect
            cur_activations[node] = activation
        if len(cur_effects) == len(effects):
            for node in effects:
                effects[node].append(cur_effects[node])
                activations[node].append(cur_activations[node])
        # del clean_outputs, cur_effects, cur_activations
        patcher.reset()
        # except Exception as e:
        #     print(f"Error processing pair: {e}")
        #     count += 1
        #     if count > 10:
        #         pdb.set_trace()
        #     continue
    # Calculate mean effects and find significant nodes
    effects = {n: torch.stack(e, dim=-1) for n, e in effects.items()}
    activations = {n: torch.stack(a, dim=1) for n, a in activations.items()}
    return [(n, effects[n], activations[n]) for n in effects]


def get_random_circuit(
    nodes: list[tuple[str, float, torch.Tensor]],
    topk: float,
    max_layers: int,
) -> nx.DiGraph:
    """
    Generate a random end-to-end subgraph of the model's computational graph.
    Args:
        nodes: List of tuples containing (node_name, importance_score, activation)
        num_nodes: Number of nodes to include in the random circuit per layer
        max_layers: Maximum number of layers in the model
        seed: Random seed for reproducibility
    Returns:
        nx.DiGraph: A directed graph representing the random circuit
    """
    cur_nodes, layer_nodes = [], get_layered_nodes_list(nodes, max_layers)

    for i in range(len(layer_nodes)):
        # Handle attention heads (q, k, v)
        num_heads = len(layer_nodes[i]["q_proj"])
        random_head_indices = torch.randperm(num_heads)[:topk]
        for head in random_head_indices:
            cur_nodes.append(layer_nodes[i]["q_proj"][head])
            cur_nodes.append(layer_nodes[i]["k_proj"][head // 4])
            cur_nodes.append(layer_nodes[i]["v_proj"][head // 4])

        # Handle output projection
        o_proj_imp = torch.stack([n[1] for n in layer_nodes[i]["o_proj"]])
        o_proj_neurons = torch.randperm(len(o_proj_imp))[:topk]
        cur_nodes.extend(
            [layer_nodes[i]["o_proj"][neuron] for neuron in o_proj_neurons]
        )

        # Handle MLP input (gate and up projections)
        random_mlp_indices = torch.randperm(len(layer_nodes[i]["gate_proj"]))[:topk]
        for neuron in random_mlp_indices:
            cur_nodes.append(layer_nodes[i]["gate_proj"][neuron])
            cur_nodes.append(layer_nodes[i]["up_proj"][neuron])

        # Handle down projection
        down_proj_imp = torch.stack([n[1] for n in layer_nodes[i]["down_proj"]])
        down_proj_neurons = torch.randperm(len(down_proj_imp))[:topk]
        cur_nodes.extend(
            [layer_nodes[i]["down_proj"][neuron] for neuron in down_proj_neurons]
        )

    # Create and clean up the circuit
    cur_circuit = circuit_to_nx(cur_nodes, max_layers, True)
    paths = nx.single_source_shortest_path(cur_circuit, "input,0,-1,input")
    unreachable_nodes = [node for node in cur_circuit.nodes if node not in paths]
    cur_circuit.remove_nodes_from(unreachable_nodes)
    cur_circuit.remove_nodes_from(
        ["input,0,-1,input", f"output,{max_layers+1},-1,output"]
    )
    return cur_circuit

def get_neurons_and_heads(
    nodes: list[tuple[str, float, torch.Tensor]],
) -> list[tuple[str, float, torch.Tensor]]:
    new_nodes = []
    for node in nodes:
        for i in range(node[1].shape[0]):
            name = node[0].split(".")[:-1] + [str(i), node[0].split(".")[-1]]
            new_nodes.append((".".join(name), node[1][i].mean(), node[2][i]))
    return new_nodes


def get_layered_nodes_list(
    nodes: list[tuple[str, float, torch.Tensor]],
    max_layers: int,
) -> list[dict[str, list[tuple[str, float, torch.Tensor]]]]:
    layer_nodes = [{} for _ in range(max_layers)]
    for node in nodes:
        node_names = node[0].split(".")
        if node_names[-1] not in layer_nodes[int(node_names[2])]:
            layer_nodes[int(node_names[2])][node_names[-1]] = []
        layer_nodes[int(node_names[2])][node_names[-1]].append(node)
    for i in range(len(layer_nodes)):
        for key in layer_nodes[i]:
            layer_nodes[i][key] = sorted(
                layer_nodes[i][key], key=lambda x: int(x[0].split(".")[2])
            )
    return layer_nodes


def get_layered_nodes_nx(
    nodes: list[str],
    max_layers: int,
) -> list[dict[str, list[str]]]:
    node_types = ["input", "output", "attn_head", "o_proj", "mlp_in", "down_proj"]
    layer_nodes = [{k: [] for k in node_types} for _ in range(max_layers)]
    for node in nodes:
        node_names = node.split(",")
        layer_nodes[int(node_names[1])][node_names[-1]].append(node)
    for i in range(len(layer_nodes)):
        for key in layer_nodes[i]:
            layer_nodes[i][key] = sorted(
                layer_nodes[i][key], key=lambda x: int(x.split(",")[2])
            )
    return layer_nodes


def find_sig_nodes(
    nodes: list[tuple[str, float, torch.Tensor]],
    topk: float,
    max_layers: int,
) -> nx.DiGraph:
    # layer_nodes: list[dict[str, list[tuple[str, torch.Tensor: n_heads 2, torch.Tensor: n_heads 2 hidden_dim]]]]
    cur_nodes, layer_nodes = [], get_layered_nodes_list(nodes, max_layers)
    pdb.set_trace()
    for i in range(len(layer_nodes)):
        # 
        attn_head_imp = torch.stack(
            [
                torch.stack([n[1] for n in layer_nodes[i]["q_proj"]]),
                torch.stack([n[1] for n in layer_nodes[i]["k_proj"]]).repeat_interleave(3, dim=-2),
                torch.stack([n[1] for n in layer_nodes[i]["v_proj"]]).repeat_interleave(
                    3, dim=-2
                ),
            ]
        )
        for head in attn_head_imp.abs().min(0)[0].topk(topk)[1]:
            pdb.set_trace()
            cur_nodes.append(layer_nodes[i]["q_proj"][0][head])
            cur_nodes.append(layer_nodes[i]["k_proj"][head // 4])
            cur_nodes.append(layer_nodes[i]["v_proj"][head // 4])
        o_proj_imp = torch.stack([n[1] for n in layer_nodes[i]["o_proj"]])
        o_proj_neurons = o_proj_imp.abs().topk(topk)[1]
        cur_nodes.extend(
            [layer_nodes[i]["o_proj"][neuron] for neuron in o_proj_neurons]
        )
        mlp_in_imp = torch.stack(
            [
                torch.stack([n[1] for n in layer_nodes[i]["gate_proj"]]),
                torch.stack([n[1] for n in layer_nodes[i]["up_proj"]]),
            ]
        )
        for neuron in mlp_in_imp.abs().min(0)[0].topk(topk)[1]:
            cur_nodes.append(layer_nodes[i]["gate_proj"][neuron])
            cur_nodes.append(layer_nodes[i]["up_proj"][neuron])
        down_proj_imp = torch.stack([n[1] for n in layer_nodes[i]["down_proj"]])
        down_proj_neurons = down_proj_imp.abs().topk(topk)[1]
        cur_nodes.extend(
            [layer_nodes[i]["down_proj"][neuron] for neuron in down_proj_neurons]
        )
    cur_circuit = circuit_to_nx(cur_nodes, max_layers, True)
    paths = nx.single_source_shortest_path(cur_circuit, "input,0,-1,input")
    unreachable_nodes = [node for node in cur_circuit.nodes if node not in paths]
    cur_circuit.remove_nodes_from(unreachable_nodes)
    cur_circuit.remove_nodes_from(
        ["input,0,-1,input", f"output,{max_layers+1},-1,output"]
    )
    return cur_circuit


def circuit_to_nx(
    nodes: list[tuple[str, float, torch.Tensor]],
    max_layers: int,
    include_in_out_nodes: bool = False,
) -> nx.DiGraph:
    G = nx.DiGraph()
    if include_in_out_nodes:
        G.add_node("input,0,-1,input", activation=-1, importance=1e10, layer=0)
        G.add_node(
            f"output,{max_layers+1},-1,output",
            activation=-1,
            importance=1e10,
            layer=max_layers + 1,
        )
    layer_nodes = get_layered_nodes_list(nodes, max_layers)
    attn_types = ["q_proj", "k_proj", "v_proj"]
    mlp_in_types = ["gate_proj", "up_proj"]
    for i in range(len(layer_nodes)):
        for comb_vals in [(attn_types, "attn_head"), (mlp_in_types, "mlp_in")]:
            for j in range(len(layer_nodes[i][comb_vals[0][0]])):
                node_names = layer_nodes[i][comb_vals[0][0]][j][0].split(".")
                if all(
                    layer_nodes[i][val][j][0].split(".")[:4] == node_names[:4]
                    for val in comb_vals[0]
                ):
                    G.add_node(
                        f"{node_names[3]},{i+1},{node_names[4]},{comb_vals[1]}",
                        activation=torch.concat(
                            [layer_nodes[i][val][j][-1] for val in comb_vals[0]], dim=1
                        ),
                        importance=torch.stack(
                            [layer_nodes[i][val][j][1] for val in comb_vals[0]]
                        ).min(),
                        layer=i + 1,
                    )
        for neuron_type in [("self_attn", "o_proj"), ("mlp", "down_proj")]:
            for j in range(len(layer_nodes[i][neuron_type[1]])):
                node_names = layer_nodes[i][neuron_type[1]][j][0].split(".")
                G.add_node(
                    f"{neuron_type[0]},{i+1},{node_names[4]},{neuron_type[1]}",
                    activation=layer_nodes[i][neuron_type[1]][j][-1],
                    importance=layer_nodes[i][neuron_type[1]][j][1],
                    layer=i + 1,
                )
    return add_edges(G, max_layers, include_in_out_nodes)


def combine_circuits(circuits: dict[str, nx.DiGraph], max_layers: int):
    combined_graph = nx.DiGraph()
    for d in circuits:
        intersection = nx.intersection(combined_graph, circuits[d])
        for n in intersection:
            if len(combined_graph.nodes[n]["activation"].shape) == 2:
                combined_graph.nodes[n]["activation"] = torch.stack(
                    [
                        combined_graph.nodes[n]["activation"],
                        circuits[d].nodes[n]["activation"],
                    ]
                )
            else:
                combined_graph.nodes[n]["activation"] = torch.concat(
                    [
                        combined_graph.nodes[n]["activation"],
                        circuits[d].nodes[n]["activation"].unsqueeze(0),
                    ]
                )
        combined_graph = nx.compose(circuits[d], combined_graph)
    for n in combined_graph:
        if len(combined_graph.nodes[n]['activation'].shape) == 2:
            combined_graph.nodes[n]['activation'] = combined_graph.nodes[n]['activation'].unsqueeze(0)
    combined_graph = add_edges(combined_graph, max_layers)
    return combined_graph


def add_edges(
    graph: nx.DiGraph,
    max_layers: int,
    include_in_out_nodes: bool = False,
) -> nx.DiGraph:
    layered_nodes = get_layered_nodes_nx(list(graph.nodes()), max_layers + 2)
    for i in range(len(layered_nodes)):
        if len(layered_nodes[i]["output"]) > 0:
            continue
        elif len(layered_nodes[i]["input"]) > 0:
            for j in range(i + 1, len(layered_nodes)):
                for attn_node in layered_nodes[j]["attn_head"]:
                    graph.add_edge(layered_nodes[i]["input"][0], attn_node)
                for mlp_in_node in layered_nodes[j]["mlp_in"]:
                    graph.add_edge(layered_nodes[i]["input"][0], mlp_in_node)
            graph.add_edge(layered_nodes[i]["input"][0], layered_nodes[-1]["output"][0])
            continue
        for attn_node in layered_nodes[i]["attn_head"]:
            for o_node in layered_nodes[i]["o_proj"]:
                graph.add_edge(attn_node, o_node)
        for o_node in layered_nodes[i]["o_proj"]:
            for mlp_in_node in layered_nodes[i]["mlp_in"]:
                graph.add_edge(o_node, mlp_in_node)
            for j in range(i + 1, len(layered_nodes)):
                for attn_node in layered_nodes[j]["attn_head"]:
                    graph.add_edge(o_node, attn_node)
                for mlp_in_node in layered_nodes[j]["mlp_in"]:
                    graph.add_edge(o_node, mlp_in_node)
            if include_in_out_nodes:
                graph.add_edge(o_node, layered_nodes[-1]["output"][0])
        for mlp_in_node in layered_nodes[i]["mlp_in"]:
            for down_node in layered_nodes[i]["down_proj"]:
                graph.add_edge(mlp_in_node, down_node)
        for down_node in layered_nodes[i]["down_proj"]:
            if include_in_out_nodes:
                graph.add_edge(down_node, layered_nodes[-1]["output"][0])
                continue
            for j in range(i + 1, len(layered_nodes)):
                for attn_node in layered_nodes[j]["attn_head"]:
                    graph.add_edge(down_node, attn_node)
                for mlp_in_node in layered_nodes[j]["mlp_in"]:
                    graph.add_edge(down_node, mlp_in_node)
    return graph


def nx_to_pyg(graph: nx.DiGraph):
    edge_idx = torch.tensor(nx.adjacency_matrix(graph).todense()).nonzero().T
    pyg_graph = Data(edge_index=edge_idx, num_nodes=graph.number_of_nodes())
    pyg_graph.activation = [graph.nodes[n]["activation"].float() for n in graph.nodes]
    pyg_graph.layer = torch.tensor(
        [graph.nodes[n]["layer"] for n in graph.nodes]
    ).float()
    return pyg_graph

def get_kmeans_partitions(comp_graph: nx.DiGraph, num_partitions: int):
    """
    Partition circuits using K-means clustering on PCA-aligned activations

    Args:
        circuits: Dictionary of circuit names to their NetworkX DiGraphs
        num_partitions: Number of desired partitions

    Returns:
        List of partitioned subgraphs
    """
    # Collect all activations and their corresponding nodes/graphs
    activations, nodes = [], []
    for node in tqdm(comp_graph.nodes()):
        pca = PCA(n_components=1)
        act = comp_graph.nodes[node]["activation"]
        if len(act.shape) == 2:
            act = act.unsqueeze(0)
        shape = list(act.shape)
        act = act.reshape(-1, act.shape[-1]).numpy()
        act = pca.fit_transform(act).reshape(tuple(shape[:-1] + [1])).mean((0, -1))
        activations.append(act)
        nodes.append(node)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_partitions, random_state=42)
    clust_labels = kmeans.fit_predict(np.array(activations).reshape(-1, activations[0].shape[0]))
    graph, partitions = comp_graph.copy(), []
    for clust_idx in range(num_partitions * 20):
        # Get nodes in this cluster
        clust_nodes = [nodes[i] for i in range(len(nodes)) if clust_labels[i] == clust_idx]
        # Only create partition if cluster has more than 2 nodes
        if len(clust_nodes) > 2:
            cur_graph = graph.subgraph(clust_nodes).copy()
            for part in nx.weakly_connected_components(cur_graph):
                if len(part) > 3:
                    partitions.append(graph.subgraph(part).copy())
                    graph.remove_nodes_from(list(part))
                    graph.remove_nodes_from(list(nx.isolates(graph)))
    partitions = sorted(partitions, key=lambda x: len(x), reverse=True)
    # If a partition is larger than 50 nodes, split into smaller partitions
    while len(partitions[0]) > 50:
        # Split it into two by finding most distant nodes
        node = random.sample(partitions[0].nodes, 1)[0]
        p1 = nx.ego_graph(partitions[0], node, radius=10)
        p2 = partitions[0].copy()
        p2.remove_nodes_from(list(p1.nodes))
        p2.remove_nodes_from(list(nx.isolates(p2)))
        # Replace largest partition with split partitions
        partitions[0] = p1
        if len(p2) > 0:
            partitions.append(p2)
        partitions = sorted(partitions, key=lambda x: len(x), reverse=True)
    for i, p in enumerate(partitions[:num_partitions]):
        for n in p:
            comp_graph.nodes[n]['partition'] = torch.tensor(i).long()
    return comp_graph


def get_random_partitions(comp_graph: nx.DiGraph, num_partitions: int):
    random_partitions, graph = [], comp_graph.copy()
    while len(random_partitions) < num_partitions and len(graph) > 0:
        node = random.sample(list(graph.nodes), 1)[0]
        possible_nodes = nx.single_source_shortest_path(graph, node, cutoff=4)
        if len(possible_nodes) < 2:
            continue
        subgraph = random.sample(list(possible_nodes), min(len(possible_nodes), 20))
        random_partitions.append([x for n in subgraph for x in possible_nodes[n]])
        graph.remove_nodes_from(random_partitions[-1])
    for i, p in enumerate(random_partitions):
        for n in p:
            comp_graph.nodes[n]["partition"] = torch.tensor(i).long()
    return comp_graph


def get_activation_partitions(comp_graph:nx.DiGraph, num_partitions: int):
    new_parts = [([k], (v.flatten().topk(5 * v.shape[0])[1] % (v.shape[1] * v.shape[2]) // v.shape[-1]).tolist()) for k,v in nx.get_node_attributes(comp_graph, "activation").items()]
    partitions, percent_diff, no_change_count = [], 0.1, 0
    with tqdm(total=100) as pbar:
        while percent_diff < 0.5 and len(new_parts) > 55:
            cur_part = new_parts.pop(0)
            merged_idx = -1
            for idx, mi in enumerate(new_parts):
                if sum([i not in mi[1] for i in cur_part[1]]) <= int(len(cur_part[1]) * percent_diff):
                    new_parts[idx] = (list(set(new_parts[idx][0] + cur_part[0])), list(set(new_parts[idx][1] + cur_part[1])))
                    merged_idx = idx
                    break
            if merged_idx == -1:
                new_parts.append(cur_part)
                no_change_count += 1
                if no_change_count > 500:
                    no_change_count = 0
                    percent_diff += 0.01
                    res = pbar.update(2.5)
            elif len(new_parts[merged_idx][1]) > 100:
                partitions.append(new_parts.pop(merged_idx))
    new_parts += partitions
    partitions, merged_idxs = [], [-1]
    while len(new_parts) > 0:
        cur_part = new_parts.pop(0)
        merged_idxs = []
        for i, p in enumerate(new_parts):
            if sum([i not in p[1] for i in cur_part[1]]) <= int(len(cur_part[1]) * 0.1):
                cur_part = (list(set(cur_part[0] + p[0])), list(set(cur_part[1] + p[1])))
                merged_idxs.append(i)
        new_parts = [n for i,n in enumerate(new_parts) if i not in merged_idxs]
        cur_subgraphs = nx.weakly_connected_components(comp_graph.subgraph(cur_part[0]))
        cur_subgraphs = [list(sub) for sub in cur_subgraphs if len(sub) > 1]
        partitions.extend([s for s in cur_subgraphs])    
    for i, p in enumerate(sorted(partitions, key=len, reverse=True)[:num_partitions]):
        for n in p:
            comp_graph.nodes[n]['partition'] = torch.tensor(i).long()
    return comp_graph


def visualize_circuit(nx_graph: nx.DiGraph, out_file: str, max_layers: int):
    label_map = {
        "input": "input",
        "output": "output",
        "attn_head": "A",
        "o_proj": "O",
        "mlp_in": "Mi",
        "down_proj": "Mo",
    }
    x_pos_map = {
        "input": 0,
        "output": 1,
        "attn_head": 1,
        "o_proj": 2,
        "mlp_in": 3,
        "down_proj": 4,
    }
    layered_nodes = get_layered_nodes_nx(list(nx_graph.nodes()), max_layers + 2)

    labels = {
        n: f'{label_map[n.split(",")[-1]]}{n.split(",")[2]}' for n in nx_graph.nodes()
    }

    nodes = []
    for i in range(1, 33):
        cur_nodes = [n for n in nx_graph.nodes if int(n.split(",")[1]) == i]
        node_map = {
            "attn_head": sorted(
                [n for n in cur_nodes if n.split(",")[-1] == "attn_head"],
                key=lambda x: int(x.split(",")[2]),
            ),
            "o_proj": sorted(
                [n for n in cur_nodes if n.split(",")[-1] == "o_proj"],
                key=lambda x: int(x.split(",")[2]),
            ),
            "mlp_in": sorted(
                [n for n in cur_nodes if n.split(",")[-1] == "mlp_in"],
                key=lambda x: int(x.split(",")[2]),
            ),
            "down_proj": sorted(
                [n for n in cur_nodes if n.split(",")[-1] == "down_proj"],
                key=lambda x: int(x.split(",")[2]),
            ),
        }
        nodes.append(node_map)
    pos = {
        n: (
            int(n.split(",")[1]) * 5 + x_pos_map[n.split(",")[-1]],
            layered_nodes[int(n.split(",")[1])][n.split(",")[-1]].index(n),
        )
        for n in nx_graph.nodes()
    }
    plt.figure(figsize=(40, 10))
    nx.draw_networkx_nodes(nx_graph, pos, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(nx_graph, pos, edge_color="gray", alpha=0.5, arrows=True)
    nx.draw_networkx_labels(nx_graph, pos, labels, font_size=8)
    ax = plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.tight_layout()
    plt.savefig(out_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()


def load_partitions(
    partitions: dict[str, torch.Tensor],
    max_layers: int,
    include_in_out_nodes: bool = False,
):
    G = nx.DiGraph()
    G.add_nodes_from(list(partitions.keys()))
    G = add_edges(G, max_layers, include_in_out_nodes)
    nx.set_node_attributes(G, partitions, "partition")
    return G