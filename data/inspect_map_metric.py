
import os 
import csv
import torch
import networkx as nx

from training.noc_dataset import NocDataset
import yaml

import matplotlib.pyplot as plt

def plot_execution_time_distributions(execution_time_dict, base_path, plot_name_prefix):
    """
    Plots separate distribution graphs for execution times for each directory in subplots,
    and exports the histogram data to CSV files for LaTeX plotting with pgfplots.
    """
    # Ensure the output folder exists
    output_path = os.path.join(base_path, "map_metric_analysis")
    os.makedirs(output_path, exist_ok=True)

    num_dirs = len(execution_time_dict)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_dirs + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)

    for i, (dir_index, execution_times) in enumerate(execution_time_dict.items()):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        # Flatten execution times for the current directory
        flattened_times = [time.item() for time in execution_times]

        # Plot histogram for the current directory
        counts, bins, patches = ax.hist(flattened_times, bins=10, alpha=0.7, color='blue', edgecolor='black')

        # Calculate bin centers for exporting to CSV
        bin_centers = [(bins[j] + bins[j + 1]) / 2 for j in range(len(bins) - 1)]

        # Export histogram data to CSV
        csv_file_path = os.path.join(output_path, f"{plot_name_prefix}_dir_{dir_index}.csv")
        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Bin Center", "Frequency"])  # Header row
            csv_writer.writerows(zip(bin_centers, counts))
        print(f"Exported histogram data for Dir {dir_index} to {csv_file_path}")

        # Annotate bars with frequency counts
        for count, bin_center in zip(counts, bin_centers):
            ax.text(bin_center, count, str(int(count)), ha='center', va='bottom', fontsize=9)

        ax.set_title(f"Dir {dir_index}")
        ax.set_xlabel('Execution Time')
        ax.set_ylabel('Frequency')

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis('off')

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_save_path = os.path.join(output_path, f"{plot_name_prefix}_plots.png")
    plt.savefig(plot_save_path)
    print(f"Saved execution time distribution plots to {plot_save_path}")


def plot_graphs_as_subplots(graphs, save_dir, save_name):

    num_graphs = len(graphs)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_graphs + cols - 1) // cols  # Calculate the number of rows needed
    node_type="task"
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)

    for idx, graph in enumerate(graphs):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        avg_clustering = nx.average_clustering(graph)
        # print(f"Graph {idx + 1}: Avg Clustering Coefficient = {avg_clustering}")

        # Filter task nodes based on the node type
        task_nodes = [node for node, attr in graph.nodes(data=True) if attr.get("type") == node_type]

        # Create a subgraph containing only the task nodes and edges between them
        task_subgraph = graph.subgraph(task_nodes)

        # Compute spring layout positions
        pos = nx.spring_layout(task_subgraph, seed=4)  # Set seed for consistent layout

        # Plot the task subgraph using spring layout
        nx.draw(
            task_subgraph,
            pos,
            with_labels=True,
            ax=ax,
            node_color="lightblue",
            edge_color="gray",
            font_weight="bold"
        )
        ax.set_title(f"Task Nodes: Graph {idx + 1}, Avg Clust: {avg_clustering}")

    # Hide unused subplots
    for idx in range(len(graphs), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis('off')

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved task nodes as subplots to {save_path}")


if __name__ == "__main__"  : 

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_test_dir", type=str, default="data/training_data/with_network/map_test" ,help="Path to the results folder")
    args = parser.parse_args()

    # map_test_directory = f"data/training_data/{args.data_path}/map_test"
    map_test_directory = args.map_test_dir

    num_dirs           = len(os.listdir(map_test_directory))

    execution_time_dict = {}
    graph_list          = []

    params = yaml.safe_load(open("training/config/params_with_network.yaml"))
    max_cycle = params["MAX_CYCLE"]

    for i in range(num_dirs): 
        dir_path = os.path.join(map_test_directory, f"{i}")
        dataset  = NocDataset(dir_path)

        graph = dataset.get_graph(0)
        graph_list.append(graph)

        execution_time_list    = []
        execution_time_dict[i] = {}

        for data in dataset: 

            data = data

            max_index       = torch.argmax(data["task"].y[:, 1])
            execution_time  = data["task"].y[max_index, 1]  #* max_cycle

            execution_time_list.append(execution_time)

        execution_time_dict[i] = execution_time_list

    plot_save_path = f"data/training_data/with_network"
    plot_execution_time_distributions(execution_time_dict, plot_save_path, "map_test_distribution.png")
    plot_graphs_as_subplots(graph_list, plot_save_path, "map_test_graphs.png")




