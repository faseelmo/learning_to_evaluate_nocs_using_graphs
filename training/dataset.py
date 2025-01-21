import os
import yaml
import torch

import networkx                 as nx

from typing                     import Union
from natsort                    import natsorted

from torch_geometric.loader     import DataLoader
from torch_geometric.data       import HeteroData, Data
from torch.utils.data           import Dataset, random_split

from data.utils                 import load_graph_from_json


class CustomDataset(Dataset):
    def __init__(self, training_data_dir, is_hetero=False, has_dependency=False, has_exit=False, has_scheduler=False, return_graph=False):
        """
        Args 
        1. is_hetero            : If True, the dataset will return a HeteroData object, else a Data object  
        2. has_scheduler_node   : If True, the dataset will connect all task nodes ('task' and 'task_depend')  
                                  to one single node.   
        3. has_task_depend      : If True, the dataset will have task_depend nodes.
                                  nodes of task_depend has predecessor connection to dependency nodes. 
                                  These nodes will need packets from the dependecy nodes to start processing.
        4. return_graph         : If True, the __get__item__ will return the graph along with the data object. 
                                  Useful for visualization.  
        """
        self._is_hetero         = is_hetero
        self._has_scheduler     = has_scheduler
        self._return_graph      = return_graph
        self._has_dependency    = has_dependency
        self._has_exit          = has_exit   # Flag to check if the graph has exit nodes

        self._file_dir              = training_data_dir
        self._training_files        = natsorted(os.listdir(training_data_dir))

        training_parameters         = yaml.safe_load(open("training/config/params_without_network.yaml"))
        self.max_generate           = training_parameters["MAX_GENERATE"]
        self.max_processing_time    = training_parameters["MAX_PROCESSING_TIME"]
        self.max_cycle              = training_parameters["MAX_CYCLE"]

    def __len__(self):
        return len(self._training_files)

    def __getitem__(self, index):
        
        file_path = os.path.join(self._file_dir, self._training_files[index])
        graph     = load_graph_from_json(file_path)

        if not self._is_hetero:
            return self._homogenous_data(graph)
        else:
            return self._heterogenous_data(graph)

    def _homogenous_data(self, graph: nx.DiGraph) -> Union[ Data, tuple[Data, tuple[dict, nx.DiGraph]] ]:
        """
        Args: 
            1. graphs       : the directed graph of idx. From data/training_data/input/task_graph_idx.json 
            2. target_data  : the output features of the graph. From data/training_data/target/task_graph_idx.json

        returns a tuple of Data and an empty dictionary (for compatibility with heterogenous data)
        """
        data = Data()

        task_input_feature  = []
        task_target_feature = []

        for idx, node_data in graph.nodes(data=True):
            generate        = node_data["generate"]         / self.max_generate
            processing_time = node_data["processing_time"]  / self.max_processing_time
            start_cycle     = node_data["start_cycle"]      / self.max_cycle
            end_cycle       = node_data["end_cycle"]        / self.max_cycle

            task_input_feature.append([generate, processing_time])
            task_target_feature.append([start_cycle, end_cycle])

        data.x = torch.tensor(task_input_feature, dtype=torch.float)
        data.y = torch.tensor(task_target_feature, dtype=torch.float)

        edge_index = []
        for src, dst in graph.edges:
            edge_index.append([src, dst])

        data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        self._do_checks(data)

        if self._return_graph:
            return (data, ({}, graph))

        else: 
            return data

    def _heterogenous_data(self, graph: nx.DiGraph) -> Union[
            HeteroData, 
            tuple[ HeteroData, tuple[dict, nx.DiGraph] ]
        ]:
        """
        Args: 
            1. graphs       : the directed graph of idx. From data/training_data/input/task_graph_idx.json 
            2. target_data  : the output features of the graph. From data/training_data/target/task_graph_idx.json

        returns a tuple of HeteroData and a dictionary containing global to local indexing

        The graph will have task and dependency nodes by default. 
        If has_wait_time is True, the graph will also have task_depend nodes.
        """
        hetero_data = HeteroData()

        task_input_feature          = []
        dependency_input_feature    = []
        exit_feature                = []

        task_target                 = [] # list of start and end cycle for each task
        exit_target                 = [] 

        global_to_local_indexing    = { "task": {}, "dependency": {}, "exit": {}, "scheduler": {} }
        
        # Convert "dependency" nodes to "task" nodes
        for node_idx, node_data in graph.nodes(data=True):
            if not self._has_dependency:
                if node_data["type"] == "dependency":
                    node_data["type"] = "task"
        
            if self._has_exit:
                successors = list(graph.successors(node_idx))
                if len(successors) == 0:
                    node_data["type"] = "exit"

            # if not self._has_task_depend:
            if node_data["type"] == "task_depend":
                node_data["type"] = "task"

        if self._has_scheduler:
            # Create a scheduler node to connect all task nodes
            scheduler_node_id   = graph.number_of_nodes()
            scheduler_edge_type = "schedules"    

            graph.add_node(scheduler_node_id, type="scheduler")
            for node_idx, node_data in graph.nodes(data=True):
                if node_data["type"] in ["task", "exit"]:
                    graph.add_edge(scheduler_node_id, node_idx, type=scheduler_edge_type)

        # Creating node features + global to local indexing
        for node_idx, node_data in graph.nodes(data=True):
            node_type = node_data["type"]

            if node_type not in hetero_data:
                hetero_data[node_type].x = []

            if node_type == "scheduler":
                global_to_local_indexing[node_type][node_idx] = len(
                    global_to_local_indexing[node_type])
                continue

            generate = node_data["generate"] / self.max_generate

            if node_type == "dependency":
                dependency_input_feature.append([generate])
                global_to_local_indexing[node_type][node_idx] = len(
                    global_to_local_indexing[node_type])
                continue 

            processing_time = node_data["processing_time"] / self.max_processing_time
            target_start_cycle = node_data["start_cycle"] / self.max_cycle
            target_end_cycle = node_data["end_cycle"] / self.max_cycle

            if node_type == "task":
                task_input_feature.append([generate, processing_time])
                task_target.append([target_start_cycle, target_end_cycle])

            if node_type == "exit":
                exit_feature.append([generate, processing_time])
                exit_target.append([target_start_cycle, target_end_cycle])

            global_to_local_indexing[node_type][node_idx] = len(
                global_to_local_indexing[node_type])

        # Flag to check if data has no task node
        task_not_emtpy = len(task_input_feature) > 0
        exit_not_empty = len(exit_feature) > 0

        # Converting list of node features to tensor
        num_features_task_node = 2

        if self._has_dependency:
            hetero_data["dependency"].x = torch.tensor( dependency_input_feature, dtype=torch.float )

        if task_not_emtpy:
            hetero_data["task"].x = torch.tensor( task_input_feature, dtype=torch.float )
            hetero_data["task"].y = torch.tensor( task_target, dtype=torch.float )
        else: 
            hetero_data["task"].x = torch.empty( (0, num_features_task_node), dtype=torch.float )
            hetero_data["task"].y = torch.empty( (0, num_features_task_node), dtype=torch.float )

        if exit_not_empty and self._has_exit:
            hetero_data["exit"].x = torch.tensor( exit_feature, dtype=torch.float )
            hetero_data["exit"].y = torch.tensor( exit_target, dtype=torch.float ) 

        elif not exit_not_empty and self._has_exit:
            hetero_data["exit"].x = torch.empty( (0, 2), dtype=torch.float )
            hetero_data["exit"].y = torch.empty( (0, 2), dtype=torch.float ) 

        if self._has_scheduler: 
            hetero_data["scheduler"].x = torch.ones( (1, 1), dtype=torch.float)

        # Creating edge indices
        generate_edge_type      = "generates"
        rev_generate_edge_type  = "requires"
        hetero_data["task", generate_edge_type, "task"].edge_index = [[], []]

        if self._has_dependency:
            # There is no task_depend node in the graph
            hetero_data["dependency", generate_edge_type, "task"].edge_index        = [[], []]
            hetero_data["task", rev_generate_edge_type, "dependency"].edge_index    = [[], []]

        if self._has_exit:
            hetero_data["task", generate_edge_type, "exit"].edge_index              = [[], []]
            hetero_data["exit", rev_generate_edge_type, "task"].edge_index          = [[], []]
            hetero_data["dependency", generate_edge_type, "exit"].edge_index        = [[], []]
            hetero_data["exit", rev_generate_edge_type, "dependency"].edge_index    = [[], []]

        if self._has_scheduler:
            rev_scheduler_edge_type = f"rev_{scheduler_edge_type}"
            hetero_data["scheduler", scheduler_edge_type, "task"].edge_index = [[], []]
            hetero_data["task", rev_scheduler_edge_type, "scheduler"].edge_index = [[], []]

            if self._has_exit:
                hetero_data["scheduler", scheduler_edge_type, "exit"].edge_index = [[], []]
                hetero_data["exit", rev_scheduler_edge_type, "scheduler"].edge_index = [[], []]

        for edge in graph.edges(data=True):
            src, dst, _ = edge

            src_type    = graph.nodes[src]["type"]
            dst_type    = graph.nodes[dst]["type"]
            do_rev      = False
            
            if src_type == "scheduler": 
                edge_type       = scheduler_edge_type
                rev_edge_type   = rev_scheduler_edge_type
                do_rev          = True

            elif src_type == "dependency": 
                edge_type       = generate_edge_type
                rev_edge_type   = rev_generate_edge_type
                do_rev          = True
            
            else: 
                edge_type = generate_edge_type

            
            src_local_index = global_to_local_indexing[src_type][src]
            dst_local_index = global_to_local_indexing[dst_type][dst]

            hetero_data[src_type, edge_type, dst_type].edge_index[0].append(src_local_index)
            hetero_data[src_type, edge_type, dst_type].edge_index[1].append(dst_local_index)

            if do_rev:
                hetero_data[dst_type, rev_edge_type, src_type].edge_index[0].append(dst_local_index)
                hetero_data[dst_type, rev_edge_type, src_type].edge_index[1].append(src_local_index)

        # Converting edge indices [list] to tensors
        for edge_type in hetero_data.edge_types:

            hetero_data[edge_type].edge_index = torch.tensor( hetero_data[edge_type].edge_index, 
                                                              dtype=torch.long ).contiguous()

        self._do_checks(hetero_data)
        
        if self._return_graph:
            return (hetero_data, (global_to_local_indexing, graph))

        else: 
            return hetero_data

    def _do_checks(self, data: Union[Data, HeteroData]) -> None:

        assert data.validate() is True, "Data is invalid"
        assert data.has_isolated_nodes() is False, "Data contains isolated nodes"
        assert data.has_self_loops() is False, "Data contains self loops"
        # assert data.is_directed() is True, "Data is not directed"


def load_data( training_data_dir, 
               batch_size          : int =32, 
               validation_split    : float =0.1,
               **kwargs ) -> tuple[DataLoader, DataLoader] :

    use_noc_dataset = kwargs.get( "use_noc_dataset", False )

    if use_noc_dataset:    
        from training.noc_dataset import NocDataset
        classify_task_nodes = kwargs.get( "classify_task_nodes", False )
        dataset            = NocDataset( training_data_dir, classify_task_nodes=classify_task_nodes )
        print(f"[load_data] Is NOC dataset: \t{use_noc_dataset}")

    else: 
        is_hetero       = kwargs.get( "is_hetero", False )
        has_scheduler   = kwargs.get( "has_scheduler", False )
        has_exit        = kwargs.get( "has_exit", False )
        has_dependency  = kwargs.get( "has_dependency", False ) 

        dataset = CustomDataset( training_data_dir  = training_data_dir, 
                                 is_hetero          = is_hetero, 
                                 has_scheduler      = has_scheduler,
                                 has_exit    = has_exit,
                                 has_dependency     = has_dependency,
                                 return_graph       = False )

    validation_size = int( validation_split * len( dataset ) )

    if validation_size == 0: # ensure at least one validation sample
        validation_size = 1  

    train_size = len( dataset ) - validation_size

    if train_size <= 0:
        raise ValueError( "Training dataset size is too small after splitting." )

    train_dataset, val_dataset = random_split( dataset, [train_size, validation_size] )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True )

    return train_loader, val_loader


if __name__ == "__main__":

    from data.utils import visualize_application

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    parser.add_argument("--is_hetero", action="store_true")
    parser.add_argument("--has_scheduler", action="store_true")
    parser.add_argument("--has_exit", action="store_true")
    parser.add_argument("--has_dependency", action="store_true")
    parser.add_argument("--show_graph", action="store_true")
    args = parser.parse_args()

    DATA_INDEX      = args.idx
    IS_HETERO       = args.is_hetero
    HAS_DEPENDENCY  = args.has_dependency   
    HAS_EXIT        = args.has_exit
    HAS_SCHEDULER   = args.has_scheduler
    SHOW_GRAPH      = args.show_graph   

    print( f"\n------Dataset Test------"                )
    print( f"Data index is          : {DATA_INDEX}"     )
    print( f"Is heterogenous graph  : {IS_HETERO}"      )
    print( f"Has dependency nodes   : {HAS_DEPENDENCY}")
    print( f"Has Exit nodes  : {HAS_EXIT}") 
    print( f"Connecting task nodes  : {HAS_SCHEDULER}"  )
    print( f"Show graph             : {SHOW_GRAPH}"     )

    DATASET_DIR = "data/training_data/without_network/train"

    dataset = CustomDataset( DATASET_DIR, 
                             is_hetero          = IS_HETERO, 
                             has_scheduler      = HAS_SCHEDULER,
                             has_exit           = HAS_EXIT,
                             has_dependency     = HAS_DEPENDENCY,
                             return_graph       = SHOW_GRAPH )   

    # from training.utils import log_hetero_data    
    # data = dataset[DATA_INDEX]
    # log_hetero_data(data)
    # print(f"{data['exit'].y}")

    def check_all_files_in_dataset():
        # Inside a def for scoping
        print( f"\n------Checking all files in the dataset------" )
        list_of_edge_type = []

        for idx, ( data, ( index_dict, graph ) ) in enumerate( dataset ):
        
            for src, dst, edge in graph.edges(data=True):
                src_type = graph.nodes[src]["type"]
                dst_type = graph.nodes[dst]["type"]
    
                edge_type = f"{src_type}_to_{dst_type}"
    
                if edge_type not in list_of_edge_type:
                    print( f"New edge type found: {edge_type}" )
                    list_of_edge_type.append( edge_type )

        print( f"[Passed] All files are valid\n" )


    if SHOW_GRAPH:
        check_all_files_in_dataset()
        exit()
        data, ( index, graph ) = dataset[DATA_INDEX]
        if IS_HETERO:
            print(f"X: {data.x_dict}")
            print(f"Edge: {data.edge_index_dict}")
        else:
            print(f"Data: {data}")
        # check_all_files_in_dataset()
        visualize_application( graph=graph )

    print( f"\n\n----------------------DataLoader Test----------------------" )

    BATCH_SIZE = 10
    print( f"\nLoading data with batch size {BATCH_SIZE}" )

    train_loader, val_loader           = load_data( training_data_dir  = DATASET_DIR, 
                                                    is_hetero          = IS_HETERO, 
                                                    batch_size         = BATCH_SIZE,
                                                    has_dependency     = HAS_DEPENDENCY,
                                                    has_exit           = HAS_EXIT,
                                                    has_scheduler      = HAS_SCHEDULER )

    print( f"DataLoader is                   {train_loader}" )
    print( f"Number of training batches:     {len(train_loader)}" )  
    print( f"Batch size:                     {train_loader.batch_size}" ) 
    print( f"Training samples                {len(train_loader) * train_loader.batch_size}" ) 
    print( f"Number of validation batches:   {len(val_loader)}" )
    
    print( f"CustomDatasert and DataLoader looking good! 😎" )
    


