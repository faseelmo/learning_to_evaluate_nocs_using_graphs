import os 
import yaml 
import random 

from src.utils  import ( graph_to_task_list, 
                         get_ordered_packet_list, 
                         simulate_application_on_pe )

from data.utils import ( generate_graph, 
                         save_graph_to_json, 
                         create_and_clear_dir, 
                         modify_graph_to_task_graph, 
                         update_graph_with_computing_list)

PARAMS = yaml.safe_load(open("training/config/params_without_network.yaml"))

def simulate(num_nodes: int, debug_mode: bool, sjf_scheduling: bool ): 

    graph = generate_graph( num_nodes )
    modify_graph_to_task_graph( graph, 
                                PARAMS['MAX_GENERATE'], 
                                PARAMS['MAX_PROCESSING_TIME'] ) # Wait time is required for ordered packet injection

    computing_list  = graph_to_task_list( graph )
    packet_list     = get_ordered_packet_list( graph )

    simulate_application_on_pe( computing_list, 
                                packet_list, 
                                debug_mode      = debug_mode, 
                                sjf_scheduling  = sjf_scheduling )

    update_graph_with_computing_list(computing_list, graph)

    if debug_mode:
        print(f"---- Debug Mode ----")
        print(f"Number of nodes: {num_nodes}")
        print(f"\nPacket List")
        print(*packet_list)

        print(f"\nComputing List")
        for idx, task in enumerate(computing_list):
            print(f"{idx}. {task}\n")

    return graph

if __name__ == "__main__":

    random.seed(0)
    training_data_dir = os.path.join("data", "training_data", "without_network", "train")
    test_data_dir     = os.path.join("data", "training_data", "without_network", "test")

    create_and_clear_dir(training_data_dir)
    create_and_clear_dir(test_data_dir)

    training_data_count = 12000
    test_split          = 1000 
    training_graphs     = []
    node_range          = (2, 6)

    DEBUG_MODE          = False
    sjf_scheduling      = True

    # Simulating
    for i in range( training_data_count ): 
        nodes = random.randint( *node_range ) 
        graph = simulate( nodes, debug_mode=DEBUG_MODE, sjf_scheduling=sjf_scheduling)
        training_graphs.append(graph)  
        print( f"\rCreating graph {i}", end='', flush=True )

    test_graphs = random.sample( training_graphs, test_split )

    for i, graph in enumerate(training_graphs):
        save_graph_to_json(graph, os.path.join(training_data_dir, f"{i}.json"))
    print("\nCreated training data.")

    print("Saving Test graphs...")
    for i, graph in enumerate(test_graphs):
        save_graph_to_json(graph, os.path.join(test_data_dir, f"{i}.json"))
