
import torch
import copy
import networkx as nx 

from noc_pysim.src.utils import get_mesh_network
from noc_pysim.src.simulator import Simulator, GraphMap

from training.train import ( get_max_latency_hetero ) 
from training.utils import hetero_graph_to_tensor, get_mask_dict_from_data

from data.utils import ( import_model_dataset_param, 
                         get_weights_from_directory,
                         assign_types_to_nodes )

def inference_unique_mapping(  mapping_list,  **kwargs ): 
    
    graph = kwargs.get("graph", None)   
    simulator = kwargs.get("simulator", None)
    model = kwargs.get("model", None)
    max_generate = kwargs.get("max_generate", 5)
    max_processing_time = kwargs.get("max_processing_time", 10)
    has_mask = kwargs.get("has_mask", False)

    graph_copy = copy.deepcopy(graph)

    task_list = simulator.graph_to_task( graph_copy )
    mapping_list = simulator.set_assigned_mapping_list( task_list, mapping_list )
    simulator.map( mapping_list )
    simulator.run()
    output_graph = get_mesh_network( mesh_size, graph_copy, mapping_list )

    hetero_data = hetero_graph_to_tensor( output_graph, max_generate, max_processing_time )
    mask_dict = get_mask_dict_from_data( hetero_data )

    if has_mask: 
        output = model( hetero_data.x_dict, hetero_data.edge_index_dict, mask_dict )
    else: 
        output = model( hetero_data.x_dict, hetero_data.edge_index_dict )

    true, pred = get_max_latency_hetero( hetero_data, output )
    simulator.clear()

    print(f"True latency: {true}, Predicted latency: {pred}")


if __name__ == "__main__": 

    model_path = "training/results/with_network/from_zip"
    model, dataset, params, result = import_model_dataset_param( model_path )
    has_mask = True
    
    if result is not None:
        best_epoch = result["best_epoch"]
    else: 
        print(f"Result is None")
        # best_epoch = 61
        raise ValueError("Result is None")


    print(f"Best Epoch: {best_epoch}")

    hidden_channels = params["HIDDEN_CHANNELS"]
    num_mpn_layers = params["NUM_MPN_LAYERS"]
    mesh_size = params["MESH_SIZE"]
    embedding_dim = params["EMBEDDING_DIM"]
    aggr_list = params["HETERO_AGGR"]   
    max_generate = params["MAX_GENERATE"]
    max_processing_time = params["MAX_PROCESSING_TIME"]

    print(f"PARAMS")
    print(f"Hidden Channels: {hidden_channels}")
    print(f"Num MPN Layers: {num_mpn_layers}")
    print(f"EMBEDDING DIM: {embedding_dim}")
    print(f"AGGR LIST: {aggr_list}")

    # Loading the model
    model = model.HeteroGNN(hidden_channels, num_mpn_layers, mesh_size, embedding_dim, aggr_list)
    model_weights_path = get_weights_from_directory( f"{model_path}/models", best_epoch )
    model.load_state_dict( torch.load( model_weights_path, weights_only=True ) )

    graph = nx.DiGraph()
    graph.add_node(0, processing_time=4)
    graph.add_node(1, processing_time=4)
    graph.add_node(2, processing_time=4)
    graph.add_node(3, processing_time=4, generate=2)

    graph.add_edge(0, 1, weight=2)
    graph.add_edge(0, 2, weight=2)
    graph.add_edge(2, 3, weight=2)
    graph.add_edge(1, 3, weight=2)

    graph = assign_types_to_nodes( graph )

    mesh_size = 4
    debug_mode = False

    simulator = Simulator ( mesh_size, mesh_size, max_cycles=1000, debug_mode=debug_mode )

    common_args = {
        "graph": graph,
        "simulator": simulator,
        "model": model,
        "max_generate": max_generate,
        "max_processing_time": max_processing_time, 
        "has_mask": has_mask
    }

    print(f"\nCondition 1 Bottom Left")
    mapping = [ GraphMap( task_id=0, assigned_pe=( 0,1 ) ), 
                GraphMap( task_id=1, assigned_pe=( 0,0 ) ),
                GraphMap( task_id=2, assigned_pe=( 1,1 ) ),
                GraphMap( task_id=3, assigned_pe=( 1,0 ) ) ]


    inference_unique_mapping( mapping, **common_args )

    print(f"\nCondition 2 Bottom Left with offset")
    mapping = [ GraphMap( task_id=0, assigned_pe=( 1,2 ) ), 
                GraphMap( task_id=1, assigned_pe=( 1,1 ) ),
                GraphMap( task_id=2, assigned_pe=( 2,2 ) ),
                GraphMap( task_id=3, assigned_pe=( 2,1 ) ) ]

    inference_unique_mapping( mapping, **common_args )

    print(f"\nCondition 3 Top Right with offset")
    mapping = [ GraphMap( task_id=0, assigned_pe=( 2,1 ) ), 
                GraphMap( task_id=1, assigned_pe=( 1,1 ) ),
                GraphMap( task_id=2, assigned_pe=( 2,2 ) ),
                GraphMap( task_id=3, assigned_pe=( 1,2 ) ) ]

    inference_unique_mapping( mapping, **common_args )

    print(f"\nCondition 4 Top Right")
    mapping = [ GraphMap( task_id=0, assigned_pe=( 3,2 ) ), 
                GraphMap( task_id=1, assigned_pe=( 3,3 ) ),
                GraphMap( task_id=2, assigned_pe=( 2,2 ) ),
                GraphMap( task_id=3, assigned_pe=( 2,3 ) ) ]

    inference_unique_mapping( mapping, **common_args )


