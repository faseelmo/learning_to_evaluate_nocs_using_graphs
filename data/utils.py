import os 
import re 
import json
import random
import numpy as np
import networkx as nx

def save_graph_to_json(graph: nx.DiGraph, filename: str):
    data = nx.node_link_data(graph)
    with open(filename, "w") as file:
        json.dump(data, file)


def load_graph_from_json(filename: str) -> nx.DiGraph:
    with open(filename, "r") as file:
        data = json.load(file)
    return nx.node_link_graph(data)


def import_model_dataset_param(model_path: str):
    import importlib.util
    import yaml 
    model_spec = importlib.util.spec_from_file_location("model", os.path.join(model_path, "model.py"))
    model_module = importlib.util.module_from_spec(model_spec)  
    model_spec.loader.exec_module(model_module)

    dataset_spec = importlib.util.spec_from_file_location("dataset", os.path.join(model_path, "dataset.py"))
    dataset_module = importlib.util.module_from_spec(dataset_spec)  
    dataset_spec.loader.exec_module(dataset_module)

    params_yaml = yaml.safe_load(open(os.path.join(model_path, "params.yaml")))
    
    results_path = os.path.join(model_path, "results.yaml")
    results_yaml = yaml.safe_load(open(results_path)) if os.path.exists(results_path) else None

    return model_module, dataset_module, params_yaml, results_yaml 


def compute_list_to_node_dict(compute_list):
    """
    Converts compute list to a dictionary with task_id as key
    and start and end cycle as value
    """
    node_dict = {}
    for task in compute_list:
        node_dict[task.task_id] = {
            "start_cycle": task.start_cycle,
            "end_cycle": task.end_cycle
        }
    return node_dict

def convert_model_output_to_compute_dict(output, indexing, max_cycle) -> dict:

    compute_dict = {}

    for key, value in output.items():

        index_key = indexing[key]
        reversed_index_key = {v: k for k, v in index_key.items()}

        for id, node in enumerate(value): 
            
            index = reversed_index_key[id]

            start_cycle = node[0].item() * max_cycle
            end_cycle   = node[1].item() * max_cycle

            compute_dict[index] = {
                "start_cycle": int(start_cycle),
                "end_cycle": int(end_cycle)
            }

    return compute_dict 

def convert_data_to_compute_dict(data, indexing, max_cycle) -> list:

    output = {}

    if data['task']: 
        output['task'] = data['task'].y

    if data['task_depend']: 
        output['task_depend'] = data['task_depend'].y

    compute_dict = convert_model_output_to_compute_dict(output, indexing, max_cycle)

    return compute_dict


def update_graph_with_computing_list(compute_list, graph: nx.DiGraph) -> nx.DiGraph: 
    """
    Updates the graph with node's start and end time 
    """

    # print(f"Graphs is {graph}")

    for task in compute_list: 
        graph.nodes[task.task_id]["start_cycle"] = task.start_cycle
        graph.nodes[task.task_id]["end_cycle"]   = task.end_cycle

    for graph_node in graph.nodes:
        if "start_cycle" not in graph.nodes[graph_node]:
            graph.nodes[graph_node]["start_cycle"] = 0
            graph.nodes[graph_node]["end_cycle"] = 0

    return graph

def generate_graph(num_nodes: int):
    """
    Generates a GNR (Growing network with reduction)
    or GNC (Growing network with copying)
    Graph will have arg "num_nodes" number of nodes
    """
    redirection_probability = random.uniform(0, 0.3)
    graph_generator = [
        lambda: nx.gnr_graph(num_nodes, redirection_probability),
        lambda: nx.gnc_graph(num_nodes),
    ]
    return random.choice(graph_generator)()


def generate_random_dag(num_nodes, edge_prob=0.1):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    # Create a random ordering
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    
    # Only add edges from earlier to later in the shuffle
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < edge_prob:
                G.add_edge(nodes[i], nodes[j])
    
    return G



def modify_graph_to_application_graph(graph: nx.DiGraph, generate_range: tuple, processing_time_range: tuple):
    """
    Adds weight attribute to the edges of the graph.
    Adds processing_time attribute to the nodes of the graph.
    Sum of the successor weights is assigned as generate attribute to the node.
    Args: 
        generate_range: tuple, range of generate values assigned to edges
        processing_time_range: tuple, range of processing time values   
    """
    for node in graph.nodes:
        processing_time                         = random.randint(*processing_time_range)
        graph.nodes[node]["processing_time"]    = processing_time
        
        successors = list(graph.successors(node))
        generate_count = 0
        for successor in successors:
            edge_weight = random.randint(*generate_range)
            graph[node][successor]["weight"] = edge_weight
            generate_count += edge_weight

        graph.nodes[node]["generate"] = generate_count

        if len(successors) == 0:
            final_node_generate_count = random.randint(*generate_range)
            graph.nodes[node]["generate"] = final_node_generate_count
            graph.nodes[node]["type"] = "exit"
            continue

        predecessors = list(graph.predecessors(node))

        if len(predecessors) == 0:
            # Assigning the node's with no incoming edges as dependency nodes
            graph.nodes[node]["type"] = "dependency"

        else: 
            graph.nodes[node]["type"] = "task"


    return graph


def assign_random_attributes(graph: nx.DiGraph, generate_range: tuple, processing_time_range: tuple):

    # Assigning random weights to the edges
    for src, dest in graph.edges:
        graph[src][dest]["weight"] = random.randint(*generate_range)

    # Assigning random processing time to the nodes
    for node in graph.nodes: 
        successor_count = len(list(graph.successors(node)))
        predecessor_count = len(list(graph.predecessors(node)))

        graph.nodes[node]["processing_time"] = random.randint(*processing_time_range)

        if successor_count == 0:
            final_node_generate_count = random.randint(*generate_range)
            graph.nodes[node]["type"] = "exit"
            graph.nodes[node]["generate"] = final_node_generate_count
            continue

        if predecessor_count == 0:
            graph.nodes[node]["type"] = "dependency"
            continue

        if predecessor_count != 0 and successor_count != 0:
            graph.nodes[node]["type"] = "task"
            continue    

    return graph


def assign_types_to_nodes(graph: nx.DiGraph):
    for node in graph.nodes:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))

        if len(predecessors) == 0:
            graph.nodes[node]["type"] = "dependency"
        elif len(successors) == 0:
            graph.nodes[node]["type"] = "exit"
        else:
            graph.nodes[node]["type"] = "task"

    return graph


def apply_matplotlib_style():
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",  # Use pdflatex
            "text.usetex": True,          # Enable LaTeX rendering
            "font.family": "serif",       # Use serif fonts
            "font.size": 20,              # Base font size for all text
            "axes.titlesize": 20,         # Font size for plot titles
            "axes.labelsize": 20,         # Font size for x and y labels
            "legend.fontsize": 20,        # Font size for legends
            "xtick.labelsize": 20,        # Font size for x-axis tick labels
            "ytick.labelsize": 20,        # Font size for y-axis tick labels
            "pgf.preamble": r"""
                \usepackage{amsmath}      % For better math rendering
                \usepackage{amssymb}      % For additional symbols
                \usepackage{siunitx}      % For units and numbers
                \def\mathdefault#1{#1}    % Prevent \mathdefault issues
            """,
        }
    )



def create_and_clear_dir(directory_path):
    import shutil
    if os.path.exists(directory_path):
        print(f"The directory {directory_path} already exists. Remove it?")
        user_input = input("Enter Yes to remove: ")
        if user_input.lower() == "yes":
            print(f"Removing {directory_path}")
            shutil.rmtree(directory_path)
    os.makedirs(directory_path)
    print(f"Created directory: {directory_path}")


def modify_graph_to_task_graph(graph: nx.DiGraph, max_generate: int, max_processing_time: int):
    """
    Add Task information as node (generate) and edge attributes (require)
    to the arg "graph"
    Wait time is used for ordered packet injection 
    packets are in ascending order of task_depend node's wait_time 
    """
    max_generate = 10
    processing_time_range = (1, max_processing_time)

    for node in graph.nodes:
        successors          = list(graph.successors(node))
        predecessors        = list(graph.predecessors(node))
        num_of_successors   = len(successors)

        generate_range          = (num_of_successors + 1, max_generate)
        random_generate_value   = random.randint(*generate_range)

        gen_split_values = get_split_value(random_generate_value, num_of_successors)

        # Condition: If the node has no incoming edges (dependency node)
        # Note: dependency nodes don't have processing time
        if len(predecessors) == 0:

            graph.nodes[node]["type"]               = "dependency"
            graph.nodes[node]["generate"]           = random_generate_value
            graph.nodes[node]["processing_time"]    = 0
            graph.nodes[node]["wait_time"]          = 0

            for successor, gen_value in zip(successors, gen_split_values):
                graph[node][successor]["weight"] = gen_value

        else:

            random_processing_time = random.randint(*processing_time_range)

            graph.nodes[node]["type"]               = "task"
            graph.nodes[node]["generate"]           = random_generate_value
            graph.nodes[node]["processing_time"]    = random_processing_time
            graph.nodes[node]["wait_time"]          = 0

            # Assigning require (edge weights) to successors by
            # splitting the generate value randomly

            gen_split_values = get_split_value(random_generate_value, num_of_successors)

            for successor, gen_value in zip(successors, gen_split_values):

                require = gen_value
                graph[node][successor]["weight"] = int(require)

        if len(predecessors) == 0 and len(successors) == 0:
            raise ValueError("Dangling node detected")

    for node in graph.nodes:
        # - Changing nodes that have 'dependency' predecessors to 'task_depend',
        #   from type 'task' to 'task_depend'
        # - Changing the wait time of the task_depend node to 
        #   max( 4 * require_value ) of its predecessors. 
        #  Note: 4 is the packet size in flit

        if graph.nodes[node]["type"] != "dependency":
            continue

        successors = list(graph.successors(node))
        max_weight = max([graph[node][successor]["weight"] for successor in successors])

        graph.nodes[node]["generate"] = max_weight

        for successor in successors:
            
            graph.nodes[successor]["type"] = "task_depend"

            # require_value   = graph[node][successor]["weight"]
            
            predecessors    = list(graph.predecessors(successor))
            require_value   = sum([graph[predecessor][successor]["weight"] for predecessor in predecessors])

            wait_time       = 4 * require_value # 4 is the packet size in flit

            current_node_wait_time = graph.nodes[successor]["wait_time"]

            if wait_time > current_node_wait_time:
                graph.nodes[successor]["wait_time"] = wait_time

    return graph


def get_split_value(generate_value: int, num_of_successors: int):
    assert (
        generate_value >= num_of_successors
    ), "generate_value must be at least as large as num_of_successors"

    base_values     = np.ones(num_of_successors, dtype=int)  # assign 1 to each successor
    remaining_value = generate_value - num_of_successors

    additional_values = np.random.multinomial(
        remaining_value, np.ones(num_of_successors) / num_of_successors
    )

    gen_split_values = base_values + additional_values
    gen_split_values = (
        gen_split_values.tolist()
    )  # Converts to list for json serialization

    for value in gen_split_values:
        assert value > 0, f"Generate split value is {value}"

    return gen_split_values

def get_compute_list_from_json(filename: str) -> dict:
    """
    Converts the node cycle information from the json file to a dictionary
    used in inspect_data.py
    """
    json_dict = json.load(open(filename))

    compute_list = {}
    for key in json_dict:
        if key == "latency":
            continue
        else: 
            compute_list[int(key)] = json_dict[key] 

    return compute_list

def get_weights_from_directory(directory: str, epoch: str):
    files = os.listdir(directory)
    epoch = str(epoch)
    for file in files:
        extracted_epoch = extract_epoch(file)
        if extracted_epoch == epoch:
            return os.path.join(directory, file)
    else: 
        raise Exception(f"File {epoch} not found in directory {directory}")

def extract_epoch(weight_path): 
    match = re.search(r'_(\d+)_(\d+)_(\w+).pth', weight_path)
    if match:
        return match.group(2)
    else:
        return None

def get_all_weights_from_directory(directory: str):
    files = os.listdir(directory)
    weights_files = []
    for file in files:
        if ".pth" in file:
            path = os.path.join(directory, file)
            weights_files.append(path)
    return weights_files

def visualize_noc_application(graph: nx.DiGraph, prediction: list = None):
    import matplotlib.pyplot  as plt
    import numpy as np
    import re

    has_prediction = False
    if prediction: 
        assert isinstance(prediction, list), "Prediction should be a list"
        has_prediction = True

    task_color_map  = { "dependency": "mediumpurple", "task": "gold", "exit": "tomato"}
    noc_color_map   = { "router": "royalblue", "pe": "mediumseagreen" }
    node_colors     = []

    for node in graph.nodes:
        if graph.nodes[node].get("type") == "task":
            node_type   = graph.nodes[node]["task_type"]
            color       = task_color_map[node_type]
            node_colors.append(color)

        else: 
            node_type   = graph.nodes[node]["type"]
            color       = noc_color_map[node_type]
            node_colors.append(color)   

    router_tilt                 = 0.4
    pe_offset                   = 0.2
    normalization_factor        = 3 
    application_graph_offset    = 4.0

    pos         = {}
    task_nodes  = [node for node, data in graph.nodes(data=True) if data.get('type') == 'task']
    task_pos    = nx.spring_layout(graph.subgraph(task_nodes), seed=0)
    pos.update(task_pos)

    # Finding normalized positions for the task nodes
    x_values = np.array([pos[0] for pos in pos.values()])
    y_values = np.array([pos[1] for pos in pos.values()])

    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    x_values = normalization_factor * (x_values - x_min) / (x_max - x_min)
    y_values = normalization_factor * (y_values - y_min) / (y_max - y_min)

    y_offset = application_graph_offset  
    y_values += y_offset

    # Updating pos with normalized task positions
    for i, node in enumerate(pos.keys()):
        pos[node] = (x_values[i], y_values[i])

    # Updating the positions of the router and PE nodes
    for node_str in graph.nodes():
        if graph.nodes[node_str].get('type') == 'router':
            x, y                = tuple( map( int, re.findall(r'\d+', node_str) ) ) 
            pos[node_str]       = ( x + router_tilt * y, y )
            pos[f"PE({x},{y})"] = ( x + (router_tilt * y) + pe_offset, y + pe_offset )

    # Custom labels for the nodes
    custom_labels = {}
    for id, node in graph.nodes(data=True): 
        label       = [f"id: {id}"]
        node_type   = node.get('type')

        if node_type == "task":
            label.append(f"True: {node.get('start_cycle', 'N/A')} -> {node.get('end_cycle', 'N/A')}")
            if has_prediction:
                start = int(prediction[id][0])
                end  = int(prediction[id][1])
                label.append(f"Pred: {start} -> {end}")

        custom_labels[id] = "\n".join(label)

    plt.figure(figsize=(10, 10))
    nx.draw(graph, 
            pos, 
            with_labels = True, 
            node_color  = node_colors, 
            node_size   = 500, 
            labels      = custom_labels, 
            font_size   = 10, 
            font_weight = 'bold', 
            edge_color  = 'gray')

    pe_nodes = [n for n in graph.nodes() if isinstance(n, str) and n.startswith("PE")]

    nx.draw_networkx_nodes(graph, pos, nodelist=pe_nodes, node_color='lightgreen', node_size=300)
    edge_labels = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

def visualize_application(
        graph: nx.DiGraph, 
        latency_value=None, 
        packet_list=None, 
        compute_list=None, 
        pred_compute_list=None):
    """
    args: 
        compute_list: list or dict. Used to display the start and end cycle (truth) of each task

    """
    import matplotlib.pyplot as plt

    node_color_map = {  "dependency": "mediumpurple", "task": "gold", "exit": "tomato", "scheduler": "mediumseagreen"}

    node_colors = [
        node_color_map.get(graph.nodes[node].get("type", "task"), "lightgreen")
        for node in graph.nodes
    ]

    seed = 0
    pos = nx.spring_layout(graph, seed=seed)
    # color_light_gray = 
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=900,
        node_color=node_colors,
        arrows=True,
        edge_color="lightgray",
    )

    if isinstance(compute_list, dict): 
        node_cycle_dict = compute_list
    else: 
        node_cycle_dict = compute_list_to_node_dict(compute_list) if compute_list else {}

    custom_labels = {}
    for node in graph.nodes:
        label_parts = [f"id: {node}"]
        if "processing_time" in graph.nodes[node]:
            label_parts.append(f"P: {graph.nodes[node]['processing_time']}")
        if "generate" in graph.nodes[node]:
            label_parts.append(f"G: {graph.nodes[node]['generate']}")
        # if "wait_time" in graph.nodes[node]:
        #     wait_time = graph.nodes[node]["wait_time"]
        #     if wait_time != 0:
        #         label_parts.append(f"W: {wait_time}")
            
        if "start_cycle" in graph.nodes[node] and "end_cycle" in graph.nodes[node]: 
            start_time = graph.nodes[node]["start_cycle"]
            end_time   = graph.nodes[node]["end_cycle"]
            label_parts.append(f"{start_time} to {end_time}")

        if node in node_cycle_dict:
            
            truth_start_cycle   = node_cycle_dict[node]['start_cycle']
            truth_end_cycle     = node_cycle_dict[node]['end_cycle']

            label_parts.append(f"T: ({truth_start_cycle}-{truth_end_cycle}) "
                               f"= {truth_end_cycle - truth_start_cycle}")

            if pred_compute_list:
                pred_start_cycle    = pred_compute_list[node]['start_cycle']
                pred_end_cycle      = pred_compute_list[node]['end_cycle']
    
                label_parts.append(f"P: ({pred_start_cycle}-{pred_end_cycle}) "
                                   f"= {pred_end_cycle - pred_start_cycle}")

        custom_labels[node] = "\n".join(label_parts)

    nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    if latency_value is not None:
        plt.text(
            0.5,
            0.05,
            f"Latency: {latency_value}",
            ha="center",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    if packet_list is not None:
        plt.text(
            0.5,
            0.01,
            f"Packet list: ←{packet_list}",
            ha="center",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    plt.show()


def does_path_contains_files(path: str):
    import os

    files = os.listdir(path)

    if len(files) > 0:
        delete_prompt = input(
            f"Path '{path}' already contains files. Do you want to delete them? (yes/no): "
        )
        if delete_prompt.lower() == "yes":
            for file in files:
                os.remove(os.path.join(path, file))
            print(f"Files in '{path}' deleted.")
        else:
            print(f"Files in '{path}' not deleted. Appending new files.")

