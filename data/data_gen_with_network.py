
import os
import yaml
import random 

from noc_pysim.src.simulator import Simulator
from data.utils              import ( generate_graph, 
                                      assign_random_attributes )

from training.utils          import does_path_exist

from noc_pysim.src.utils     import ( get_mesh_network, 
                                      save_graph_to_json, 
                                      load_graph_from_json )

def simulate(graph, map_count, params):

    graph_list = []
    mesh_size = params['MESH_SIZE']
    max_cycle = params['MAX_CYCLE']
    
    for i in range(map_count): 

        sim = Simulator( num_rows=mesh_size, 
                         num_cols=mesh_size, 
                         max_cycles= max_cycle, 
                         debug_mode=False )

        task_list    = sim.graph_to_task(graph)
        mapping_list = sim.get_random_mapping(task_list)

        sim.map(mapping_list)
        latency = sim.run()

        output_graph = get_mesh_network(mesh_size, graph, mapping_list)
        graph_list.append(output_graph)

    return graph_list


def create_original_graph(count, node_range, save_dir, params): 
    """
    These graphs will be used to generate training data in data_gen_map 
    """

    generate_range = (params['MIN_GENERATE'], params['MAX_GENERATE'])
    processing_range = (params['MIN_PROCESSING_TIME'], params['MAX_PROCESSING_TIME'])

    for i in range(count): 
        num_nodes = random.randint(*node_range)
        graph = generate_graph(num_nodes)
        graph = assign_random_attributes(graph, generate_range, processing_range)
        save_graph_to_json(graph, os.path.join(save_dir, f"{i}.json"))
        print(f"\rCreating Original Graph {i}", end='', flush=False)

    print(f"\nTotal original graphs created: {count}\n")


def create_test_graphs(count, node_range, save_dir, params):
    generate_range = (params['MIN_GENERATE'], params['MAX_GENERATE'])
    processing_range = (params['MIN_PROCESSING_TIME'], params['MAX_PROCESSING_TIME'])

    for i in range(count):
        num_nodes = random.randint(*node_range)
        graph = generate_graph(num_nodes)
        graph = assign_random_attributes(graph, generate_range, processing_range)
        graph_list = simulate(graph, map_count=1, params=params)
        save_graph_to_json(graph_list[0], os.path.join(save_dir, f"{i}.json"))
        print(f"\rCreating Test Graph {i}", end='', flush=False)

    print(f"\nTotal test graphs created: {count}\n")

    

def create_train_data(map_count, save_dir, params, batch_size=50000):
    batch_graphs = []
    total_saved = 0

    unmapped_graph_path = "data/training_data/with_network/unmapped_train"

    files = os.listdir(unmapped_graph_path)

    for i, file in enumerate(files): 
        graph = load_graph_from_json(os.path.join(unmapped_graph_path, file))
        graph_list = simulate(graph, map_count, params)

        batch_graphs.extend(graph_list)

        if len(batch_graphs) >= batch_size:
            print(f"\nSaving batch of {len(batch_graphs)} graphs to disk...")
            save_data(batch_graphs, save_dir, start_idx=total_saved)
            total_saved += len(batch_graphs)
            batch_graphs.clear()

        print(f"\rCreating Train Graph {(i + 1) * map_count}", end='', flush=False)

    if batch_graphs:
        print(f"\nSaving batch of {len(batch_graphs)} graphs to disk...")
        save_data(batch_graphs, save_dir, start_idx=total_saved)

    print(f"\nTotal training graphs created: {total_saved + len(batch_graphs)}\n")


def save_data(graph_list, data_dir, start_idx=0):
    for i, graph in enumerate(graph_list):
        save_graph_to_json(graph, os.path.join(data_dir, f"{start_idx + i}.json"))


if __name__ == "__main__": 


    random.seed(0)

    graph_count         = 12000   
    training_map_count  = 6  
    print(f"Total training data needed: {graph_count * training_map_count}")

    test_data_count     = 1000

    node_range          = (2, 6)
    batch_size          = 50000

    params = yaml.safe_load(open("training/config/params_with_network.yaml"))  

    training_data_dir   = os.path.join( params['DATA_DIR'], "train" )
    test_data_dir       = os.path.join( params['DATA_DIR'], "test" )
    unmapped_graph_dir  = os.path.join( params['DATA_DIR'], "unmapped_train" )

    does_path_exist( test_data_dir )
    does_path_exist( unmapped_graph_dir )
    does_path_exist( training_data_dir  )

    create_original_graph( count=graph_count, 
                           node_range=node_range, 
                           save_dir=unmapped_graph_dir, 
                           params=params )

    create_train_data( map_count=training_map_count,
                       save_dir=training_data_dir,
                       params=params,
                       batch_size=batch_size )

    create_test_graphs( count=test_data_count,
                        node_range=node_range,
                        save_dir=test_data_dir,
                        params=params )


