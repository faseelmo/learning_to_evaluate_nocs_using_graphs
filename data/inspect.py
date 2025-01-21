
import os 
import yaml
import torch
import argparse
import importlib.util 

from data.utils import load_graph_from_json, get_weights_from_directory, visualize_application, visualize_noc_application, convert_model_output_to_compute_dict, convert_data_to_compute_dict

from training.utils import get_max_latency_truth_pred

if __name__ == "__main__" :

    parser  = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    parser.add_argument("--with_network", action="store_true", help="Use the model with network")   
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")

    args    = parser.parse_args()

    model_spec = importlib.util.spec_from_file_location("model", os.path.join(args.model_path, "model.py"))
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    dataset_spec = importlib.util.spec_from_file_location("dataset", os.path.join(args.model_path, "dataset.py"))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_module)

    index = args.idx

    params = yaml.safe_load(open(os.path.join(args.model_path, "params.yaml"), "r"))

    max_cycle   = params["MAX_CYCLE"]
    channel     = params["HIDDEN_CHANNELS"]
    num_layers  = params["NUM_MPN_LAYERS"]
    
    if args.with_network:
        dataset  = dataset_module.NocDataset("data/training_data/with_network/test", False)

    else:
        is_hetero        = params["IS_HETERO"].strip().lower()       == "true" 
        has_dependency   = params["HAS_DEPENDENCY"].strip().lower()  == "true" 
        has_task_depend  = params["HAS_TASK_DEPEND"].strip().lower() == "true" 
        has_scheduler    = params["HAS_SCHEDULER"].strip().lower()   == "true" 

        dataset         = dataset_module.CustomDataset( "data/training_data/without_network/test", 
                                                        is_hetero, 
                                                        has_scheduler,
                                                        has_task_depend, 
                                                        has_dependency,
                                                        True ) 
        data, graph_stuff   = dataset[index]
        model               = model_module.MPNHetero( channel, num_layers, "graphconv", data.metadata() ) 

    model_path  = os.path.join( args.model_path, "models" )
    weight_path = get_weights_from_directory( model_path, args.epoch )

    model.load_state_dict( torch.load(weight_path, weights_only=True) )

    if args.with_network:
        graph_path = f"data/training_data/with_network/train/{index}.json"
        graph = load_graph_from_json( graph_path)
        visualize_noc_application(graph)

    else: 

        data, graph_stuff = dataset[index]

        indexing = graph_stuff[0]
        graph    = graph_stuff[1]

        output            = model(data.x_dict, data.edge_index_dict)
        pred_compute_dict = convert_model_output_to_compute_dict(output, indexing, max_cycle)
        true_compute_dict = convert_data_to_compute_dict(data, indexing, max_cycle)

        true_max_latency, pred_max_latency = get_max_latency_truth_pred(data, output)
        true_max_latency = int(true_max_latency.item() * max_cycle)
        pred_max_latency = int(pred_max_latency * max_cycle)

        print(f"True max latency is {true_max_latency} and pred max latency is {pred_max_latency}")
        visualize_application(graph=graph, compute_list=true_compute_dict, pred_compute_list=pred_compute_dict)





    



