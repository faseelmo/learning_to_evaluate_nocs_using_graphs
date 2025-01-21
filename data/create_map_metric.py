
import os 
import yaml 
import random
import subprocess
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from training.utils import does_path_exist
from data.utils import ( generate_graph, 
                         modify_graph_to_application_graph, 
                         assign_random_attributes, 
                         visualize_noc_application, 
                         save_graph_to_json ) 

from src.simulator import Simulator
from src.utils import get_mesh_network

PARAMS = yaml.safe_load(open("training/config/params_with_network.yaml"))

if __name__ == "__main__":

    random.seed(1)

    count = 10
    maps_per_count = 100
    node_range = (2, 5)  # Range for node count in graphs
    max_graphs_per_node_count = 4  # Max graphs to store per node count

    map_data_dir = os.path.join(PARAMS['DATA_DIR'], "map_test")
    does_path_exist(map_data_dir)

    iso_graph_list = []  # List to store non-isomorphic graphs
    node_count_dict = {}  # To count graphs per node count

    while True: 
        node_count = random.randint(*node_range)
        graph = generate_graph(node_count)


        avg_clustering = nx.average_clustering(graph)
        if avg_clustering > 0.4: 
            continue


        is_unique = True
        for existing_graph in iso_graph_list:
            if nx.is_isomorphic(graph, existing_graph):
                is_unique = False
                break

        if is_unique:
            if node_count_dict.get(node_count, 0) < max_graphs_per_node_count:
                iso_graph_list.append(graph)
                node_count_dict[node_count] = node_count_dict.get(node_count, 0) + 1

        if len(iso_graph_list) >= count:
            break

    # for i, graph in enumerate(iso_graph_list):
    #     nx.draw(graph, with_labels=True)
    #     avg_clustering = nx.average_clustering(graph)
    #     print(f"\nGraph {i}: avg_clustering: {avg_clustering}")
    #     plt.show()

    print(f"Total unique graphs: {len(iso_graph_list)}")

    idx = 0
    map_latencies = []
    min_latency_range = 100
    mesh_size = PARAMS['MESH_SIZE']
    max_cycles = PARAMS['MAX_CYCLE']

    # print(map_data_dir)
    # exit()

    while True: 

        # print(f"Graph index: {idx}")
        map_graph_list = []
        graph = iso_graph_list[idx]
        generate_range = ( PARAMS['MIN_GENERATE'] , PARAMS['MAX_GENERATE'] )
        processing_range = ( PARAMS['MIN_PROCESSING_TIME'] , PARAMS['MAX_PROCESSING_TIME'] )
        graph = assign_random_attributes( graph, generate_range, processing_range )

        for i in range(maps_per_count): 

            sim = Simulator( num_rows=mesh_size, 
                             num_cols=mesh_size,
                             max_cycles=max_cycles )

            task_list = sim.graph_to_task(graph)
            mapping_list = sim.get_random_mapping(task_list)
            sim.map(mapping_list)
            latency = sim.run()

            output_graph = get_mesh_network( mesh_size, graph, mapping_list )
            map_graph_list.append(output_graph)
            map_latencies.append(latency)

        latency_range = np.max(map_latencies) - np.min(map_latencies)

        # print(f"Latency range: {latency_range}")
        if latency_range > min_latency_range: 
            avg_clustering = nx.average_clustering(graph)
            is_dag = nx.is_directed_acyclic_graph(graph)
            print(f"Created maps for graph {idx} with avg clustering: {avg_clustering}, is_dag: {is_dag}")  

            save_dir = os.path.join(map_data_dir, f"{idx}")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for i, map_graph in enumerate(map_graph_list): 
                path = os.path.join(save_dir, f"{i}.json")
                save_graph_to_json( map_graph, path)

            idx += 1
            if idx >= len(iso_graph_list): 
                break

    command = [
        "python3", "-m", "data.inspect_map_metric",
        "--map_test_dir", map_data_dir 
    ]

    subprocess.run(command)








