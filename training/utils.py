
def log_hetero_data(data) -> None:

        from torch_geometric.data import HeteroData
        assert isinstance(data, HeteroData), "Data is not of type HeteroData"

        print(f"\n---HeteroData---")
        print(f"Data.x: {data.x_dict}")
        print(f"\nEdges")
        for edge_index in data.edge_index_dict:
            print(f"\nEdge index: {edge_index} \n{data.edge_index_dict[edge_index]}")


def does_model_dir_exit(dir_path) -> None:
    import os
    model_path = os.path.join(dir_path, "models")   
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Folder '{model_path}' created.")


def does_path_exist(dir_path) -> None:
    import os
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Folder '{dir_path}' created.")
    else:
        print(f"Folder '{dir_path}' already exists.")
        continue_prompt = input("Do you want to continue? (yes/no): ")
        if continue_prompt.lower() != "yes":
            exit()


def copy_file(src_path, dest_path):
    import shutil

    shutil.copy2(src_path, dest_path)
    # print(f"Copied {src_path} to {dest_path}")


def print_parameter_count(model):
    num_params = sum(
        p.numel() 
        for p in model.parameters() 
        if p.requires_grad
    )
    print(f"Number of parameters: {num_params}")
    return num_params


def get_metadata(dataset_path, **kwargs): 
    
    use_noc_dataset = kwargs.get( "use_noc_dataset", False )

    if use_noc_dataset:
        from training.noc_dataset import NocDataset
        dataset  = NocDataset(dataset_path)
        metadata = dataset[0].metadata()

    else: 
        is_hetero       = kwargs.get( "is_hetero", False )
        has_scheduler   = kwargs.get( "has_scheduler", False )
        has_dependency  = kwargs.get( "has_dependency", False ) 
        has_exit        = kwargs.get( "has_exit", False )


        # print(f"Has task depend: {has_task_depend}")

        # print(f"Fetching metadata for dataset without_network")
        from training.dataset import CustomDataset
        dataset = CustomDataset( dataset_path, 
                                 is_hetero          = is_hetero, 
                                 has_scheduler      = has_scheduler, 
                                 has_exit           = has_exit,
                                 has_dependency     = has_dependency,
                                 return_graph       = False )

        metadata = dataset[0].metadata()

    return metadata

def initialize_model(model, dataloader, device):
    """Initialize the model by performing a dummy forward pass."""
    import torch
    from torch_geometric.data import Data, HeteroData
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = next(iter(dataloader))
        data = data.to(device)  # Ensure data is on the correct device
        
        if isinstance(data, HeteroData):
            
            mask_dict = get_mask_dict_from_data(data)
            if len(mask_dict) > 0:
                model(data.x_dict, data.edge_index_dict, mask_dict)
            else:
                model(data.x_dict, data.edge_index_dict)

        else:
            model(data.x, data.edge_index)  

    # Verify all parameters are initialized
    for name, param in model.named_parameters():
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            raise ValueError(f"Parameter {name} is still uninitialized.")
    # print(f"Model initialized")

def get_mask_dict_from_data(data):

    has_router_mask = all(hasattr(data, attr) for attr in ['corner_router_mask', 'border_router_mask', 'normal_router_mask'])
    has_pe_mask = all(hasattr(data, attr) for attr in ['corner_pe_mask', 'border_pe_mask', 'normal_pe_mask'])

    mask_dict = {}

    if has_router_mask:
        mask_dict.update({
            'corner_router_mask': data.corner_router_mask,
            'border_router_mask': data.border_router_mask,
            'normal_router_mask': data.normal_router_mask
        })

    if has_pe_mask:
        mask_dict.update({
            'corner_pe_mask': data.corner_pe_mask,
            'border_pe_mask': data.border_pe_mask,
            'normal_pe_mask': data.normal_pe_mask
        })

    if not has_router_mask and not has_pe_mask:
        mask_dict = {}

    return mask_dict


def plot_and_save_loss(train_loss, valid_loss, test_metric, save_path):
    import matplotlib.pyplot as plt
    import pickle

    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_yscale("log")
    ax1.plot(epochs, train_loss, label="Training Loss")
    ax1.plot(epochs, valid_loss, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(epochs, test_metric, "g-", label="Kendall's Tau")
    ax2.set_ylabel("Kendall's Tau")
    ax2.tick_params(axis="y")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title("Training and Validation Loss (Log Scale) with Kendall's Tau")
    plt.savefig(
        f"{save_path}/plot.png"
    )
    plt.clf()

    loss_dict = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "kendalls_tau": test_metric,
    }
    with open(
        f"{save_path}/loss.pkl", "wb"
    ) as file:
        pickle.dump(loss_dict, file)


from itertools import combinations 

def adjusted_kendalls_tau(x: list, y: list, t_x: int=0, t_y: int=0) -> float:
    """
    Args:
        x, y        : lists of values
        t_x, t_y    : threshold values for x and y  
    """
    n = len(x)
    if n != len(y):
        raise ValueError("The two lists must have the same length.")

    C, D = 0.0, 0.0 

    for (i, j) in combinations(range(n), 2): 
        dx = x[i] - x[j]
        dy = y[i] - y[j]

        if ( dx > t_x and dy > t_y ) or ( dx < -t_x and dy < -t_y ): # Concordant
            C += 1

        elif ( dx > t_x and dy < -t_y ) or ( dx < -t_x and dy > t_y ): # Discordant
            D += 1
        
        else: # Tied
            if abs(dx) <= t_x and abs(dy) <= t_y: # Both pairs are effectively equal
                C += 1
            else: 
                D += 0.25

    total_pairs = n * (n - 1) / 2
    tau = (C - D) / total_pairs

    return round(tau, 2)
        


def hetero_graph_to_tensor(graph, max_generate, max_processing_time):
    from torch_geometric.data import HeteroData
    import torch 
    # same code from training/noc_dataset.py __getitem__ method


    data = HeteroData()

    task_feature        = []
    exit_feature        = []
    dependency_feature  = []

    task_target = []
    exit_target = []

    border_router_mask = []
    corner_router_mask = []
    normal_router_mask = []

    final_latency = 0
    # Each node type has its own indexing
    global_to_local_indexing = { "router": {}, "pe": {}, "task": {}, "dependency": {}, "exit": {} }

    for node_id, node_data in graph.nodes(data=True): 
        node_type = node_data["type"]
        if node_type == "task": 
            task_type       = node_data["task_type"]
            generate        = node_data["generate"] / max_generate 
            start_cycle     = node_data["start_cycle"] # / self.max_cycle
            end_cycle       = node_data["end_cycle"]   # / self.max_cycle
            processing_time = node_data["processing_time"] / max_processing_time

            if task_type == "exit": 
                if end_cycle > final_latency:
                    final_latency = end_cycle
            task_feature.append([generate, processing_time])
            task_target.append([start_cycle, end_cycle])

        elif node_type == "router":

            num_edges = len(list(graph.edges(node_id)))
            if num_edges == 5: 
                normal_router_mask.append(1)
                corner_router_mask.append(0)
                border_router_mask.append(0)

            elif num_edges == 4:
                border_router_mask.append(1)
                corner_router_mask.append(0)
                normal_router_mask.append(0)

            elif num_edges == 3:
                corner_router_mask.append(1)
                border_router_mask.append(0)
                normal_router_mask.append(0)

        global_to_local_indexing[node_type][node_id] = len(global_to_local_indexing[node_type])

    # Task (depend, task_depend, task) features
    # Creating the input and target tensors
    has_task_node = len(task_feature) > 0
    num_features_task_node = 2

    if has_task_node:
        data["task"].x = torch.tensor( task_feature, dtype=torch.float )
        data["task"].y = torch.tensor( task_target, dtype=torch.float ) 

    else: 
        data["task"].x = torch.empty( (0, num_features_task_node), dtype=torch.float )
        data["task"].y = torch.empty( (0, num_features_task_node), dtype=torch.float )

    # Creating fake/empty inputs for routers and pes
    # This doesnt matter anyways since this feature is getting replaced with node embeddings
    num_elements = len(global_to_local_indexing["pe"])
    data["router"].x    = torch.ones( num_elements, 1, dtype=torch.float )
    data["pe"].x        = torch.ones( num_elements, 1, dtype=torch.float )

    # Creating the edge index tensor
    task_edge       = "generates_for"
    rev_task_edge   = "requires"
    task_pe_edge    = "mapped_to"
    pe_task_edge    = "rev_mapped_to"
    router_edge     = "link"
    router_pe_edge  = "interface"
    pe_router_edge  = "rev_interface"

    # NoC edges
    data["router",  router_edge,    "router"].edge_index = [ [], [] ]
    data["router",  router_pe_edge, "pe"].edge_index     = [ [], [] ]
    data["pe",      pe_router_edge, "router"].edge_index = [ [], [] ]

    # Task edges
    data["task", task_edge,     "task"].edge_index = [ [], [] ] 
    data["task", rev_task_edge, "task"].edge_index = [ [], [] ] 

    # Map edges
    data["task", task_pe_edge, "pe"].edge_index   = [ [], [] ]
    data["pe",   pe_task_edge, "task"].edge_index = [ [], [] ]
    
    for edge in graph.edges(data=True):
        src_node, dst_node, _ = edge
        src_type = graph.nodes[src_node]["type"]
        dst_type = graph.nodes[dst_node]["type"]
        do_rev = False
        if src_type == "task" and dst_type == "task": 
            edge_type       = task_edge   
            rev_edge_type   = rev_task_edge 
            do_rev          = True
            
        elif src_type == "task" and dst_type == "pe": 
            edge_type       = task_pe_edge
            rev_edge_type   = pe_task_edge
            do_rev          = True

        elif src_type == "router" and dst_type == "router":
            edge_type = router_edge

        elif src_type == "router" and dst_type == "pe":
            edge_type = router_pe_edge

        elif src_type == "pe" and dst_type == "router":
            edge_type = pe_router_edge  

        else:
            raise ValueError(f"Invalid edge type from {src_type} to {dst_type}")

        src_local_index = global_to_local_indexing[src_type][src_node]
        dst_local_index = global_to_local_indexing[dst_type][dst_node]  

        data[src_type, edge_type, dst_type].edge_index[0].append(src_local_index)
        data[src_type, edge_type, dst_type].edge_index[1].append(dst_local_index)

        if do_rev : 
            # Reversee edge for task-pe and task-task edges
            # print(f"Reversing edge from {src_type} to {dst_type}")
            data[dst_type, rev_edge_type, src_type].edge_index[0].append(dst_local_index)
            data[dst_type, rev_edge_type, src_type].edge_index[1].append(src_local_index)   

    for edge_type in data.edge_types: 
        data[edge_type].edge_index = torch.tensor(data[edge_type].edge_index, dtype=torch.long).contiguous()

    data.y = torch.tensor([final_latency], dtype=torch.float)
    data.corner_router_mask = torch.tensor(corner_router_mask, dtype=torch.float)
    data.border_router_mask = torch.tensor(border_router_mask, dtype=torch.float)   
    data.normal_router_mask = torch.tensor(normal_router_mask, dtype=torch.float)
    
    return data 