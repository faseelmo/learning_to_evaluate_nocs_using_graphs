
import os 
import re
import yaml 
import torch

from natsort import natsorted

from data.utils import load_graph_from_json

from torch.utils.data import Dataset    
from torch_geometric.data import HeteroData

class NocDataset(Dataset): 
    def __init__(self, training_data_dir, classify_task_nodes: bool = False): 

        self._file_dir              = training_data_dir  
        self._training_files        = natsorted(os.listdir(training_data_dir))
        self._classify_task_nodes   = classify_task_nodes   

        training_parameters         = yaml.safe_load(open("training/config/params_with_network.yaml"))
        self.max_generate           = training_parameters["MAX_GENERATE"]
        self.max_processing_time    = training_parameters["MAX_PROCESSING_TIME"]
        self.max_cycle              = training_parameters["MAX_CYCLE"]

        self.return_graph           = False    

    def __len__(self) -> int: 
        return len(self._training_files)

    def get_graph(self, index: int):
        file_path = os.path.join(self._file_dir, self._training_files[index])
        graph = load_graph_from_json(file_path)
        return graph    

    def __getitem__(self, index: int):

        file_path  = os.path.join(self._file_dir, self._training_files[index])
        graph      = load_graph_from_json(file_path)

        # from data.utils import visualize_noc_application
        # visualize_noc_application(graph)

        data = HeteroData()

        task_feature        = []
        exit_feature        = []
        dependency_feature  = []

        task_target = []
        exit_target = []

        border_router_mask = []
        corner_router_mask = []
        normal_router_mask = []

        border_pe_mask = []
        corner_pe_mask = []
        normal_pe_mask = []

        final_latency = 0

        # Each node type has its own indexing
        global_to_local_indexing = { "router": {}, "pe": {}, "task": {}, "dependency": {}, "exit": {} }

        for node_id, node_data in graph.nodes(data=True): 
            node_type = node_data["type"]

            if node_type == "task": 
                task_type       = node_data["task_type"]
                generate        = node_data["generate"] / self.max_generate 
                start_cycle     = node_data["start_cycle"] # / self.max_cycle
                end_cycle       = node_data["end_cycle"]   # / self.max_cycle
                processing_time = node_data["processing_time"] / self.max_processing_time

                if task_type == "exit": 
                    if end_cycle > final_latency:
                        final_latency = end_cycle

                if self._classify_task_nodes:
                    if task_type == "dependency":
                        dependency_feature.append([generate])

                    elif task_type == "task":
                        task_feature.append([generate, processing_time])
                        task_target.append([start_cycle, end_cycle])   

                    elif task_type == "exit":
                        exit_feature.append([generate, processing_time])
                        exit_target.append([start_cycle, end_cycle])

                    else: 
                        raise ValueError(f"Invalid task type {task_type}")
                    
                    node_type = task_type   

                else: 
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

            elif node_type == "pe":
                # have to do this separately to match the indexing
                successors = list(graph.successors(node_id))
                for successor in successors: 
                    if graph.nodes[successor]["type"] == "router":
                        num_router_edges = len(list(graph.edges(successor)))
                        if num_router_edges == 5: 
                            normal_pe_mask.append(1)
                            corner_pe_mask.append(0)
                            border_pe_mask.append(0)
                        elif num_router_edges == 4:
                            border_pe_mask.append(1)
                            corner_pe_mask.append(0)
                            normal_pe_mask.append(0)
                        elif num_router_edges == 3:
                            corner_pe_mask.append(1)
                            border_pe_mask.append(0)
                            normal_pe_mask.append(0)
                        
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

        if self._classify_task_nodes:
            data["dependency"].x = torch.tensor( dependency_feature, dtype=torch.float )
            data["exit"].x = torch.tensor( exit_feature, dtype=torch.float )
            data["exit"].y = torch.tensor( exit_target, dtype=torch.float )

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

        if self._classify_task_nodes:
            data["dependency",  task_edge,      "task"].edge_index       = [ [], [] ]
            data["dependency",  task_edge,      "exit"].edge_index             = [ [], [] ] 
            data["task",        rev_task_edge,  "dependency"].edge_index = [ [], [] ]
            data["task",        task_edge,      "exit"].edge_index       = [ [], [] ]
            data["exit",        rev_task_edge,  "task"].edge_index       = [ [], [] ]

        # Map edges
        data["task", task_pe_edge, "pe"].edge_index   = [ [], [] ]
        data["pe",   pe_task_edge, "task"].edge_index = [ [], [] ]
        
        if self._classify_task_nodes:
            data["exit",        task_pe_edge, "pe"].edge_index = [ [], [] ]
            data["dependency",  task_pe_edge, "pe"].edge_index = [ [], [] ]

            data["pe", pe_task_edge, "exit"].edge_index  = [ [], [] ]   
            data["pe", pe_task_edge, "dependency"].edge_index   = [ [], [] ]

        for edge in graph.edges(data=True):

            src_node, dst_node, _ = edge
            src_type = graph.nodes[src_node]["type"]
            dst_type = graph.nodes[dst_node]["type"]

            do_rev = False

            if src_type == "task" and dst_type == "task": 
                edge_type       = task_edge   
                rev_edge_type   = rev_task_edge 
                if self._classify_task_nodes:
                    do_rev          = False
                else: 
                    do_rev          = True
                
                if self._classify_task_nodes:
                    src_type = graph.nodes[src_node]["task_type"]
                    dst_type = graph.nodes[dst_node]["task_type"]

            elif src_type == "task" and dst_type == "pe": 
                edge_type       = task_pe_edge
                rev_edge_type   = pe_task_edge
                do_rev          = True

                if self._classify_task_nodes:
                    src_type = graph.nodes[src_node]["task_type"]

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

        data.corner_pe_mask = torch.tensor(corner_pe_mask, dtype=torch.float)   
        data.border_pe_mask = torch.tensor(border_pe_mask, dtype=torch.float)   
        data.normal_pe_mask = torch.tensor(normal_pe_mask, dtype=torch.float)
        
        self._do_checks(data)

        if self.return_graph:
            return data, graph

        return data 

    def _extract_coordinates(self, element_str: str) -> tuple: 
        x, y = tuple(map(int, re.findall(r'\d+', element_str)))
        return x, y


    def _do_checks( self, data: HeteroData ) -> None:

        assert data.validate() is True, "Data is invalid"
        assert data.has_isolated_nodes() is False, "Data contains isolated nodes"
        assert data.has_self_loops() is False, "Data contains self loops"

if __name__ == "__main__": 

    from training.dataset import load_data
    from training.model_with_network import  HeteroGNN
    from data.utils import visualize_noc_application

    import random 
    import torch
    random.seed(0)
    torch.manual_seed(0)

    idx = 0 # 3 has all three task types 

    dataset = NocDataset( "data/training_data/with_network_4x4/test" )
    print(f"Length of dataset is {len(dataset)}")
    data = dataset[idx]  
    print(f"data is {data}")

    print(f"Final latency is {data.y}") 
    
    print(f"Corner router mask is \t{data.corner_router_mask}")
    print(f"Corner pe mask is \t{data.corner_pe_mask}")
    print(f"")
    print(f"Border router mask is \t{data.border_router_mask}")
    print(f"Border pe mask is \t{data.border_pe_mask}")
    print(f"")
    print(f"Normal router mask is \t{data.normal_router_mask}")
    print(f"Normal pe mask is \t{data.normal_pe_mask}")


    
    # for key, value in data.edge_index_dict.items(): 
    #     print(f"{key} edge index is \n{value}")




