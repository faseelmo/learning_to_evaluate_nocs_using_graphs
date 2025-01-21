import os 
import yaml 
import torch
import argparse 
import subprocess
import numpy as np
import pickle 
import importlib.util 

from scipy.stats    import kendalltau
from data.utils     import ( get_weights_from_directory, 
                             get_all_weights_from_directory, 
                             extract_epoch )

from training.train import get_max_latency_hetero
from training.utils import print_parameter_count, get_mask_dict_from_data

def get_best_application_score(weight_path): 
    grand_parent_dir = os.path.dirname(os.path.dirname(weight_path))
    loss_file_path = os.path.join(grand_parent_dir, "loss.pkl")

    with open(loss_file_path, "rb") as f:
        loss = pickle.load(f)

    kendalls_tau = loss["kendalls_tau"]

    # for i, tau in enumerate(kendalls_tau): 
    #     print(f"Epoch: {i+1}\tTau: {tau}")

    best_epoch = np.argmax(kendalls_tau)
    best_score = kendalls_tau[best_epoch]

    return best_score, best_epoch   


def get_application_score(weight_path, total_epochs, best_epoch):
    grand_parent_dir = os.path.dirname(os.path.dirname(weight_path))
    file_path = os.path.join(grand_parent_dir, "loss.pkl")

    with open(file_path, "rb") as f:
        loss = pickle.load(f)

    best_epoch = int(best_epoch) - 1

    if best_epoch == total_epochs:
        # for some reason the last epoch is saved eas EPOCH+1.
        # I dont want to fix it now. 
        best_epoch = len(loss["kendalls_tau"]) - 1

    application_score = loss["kendalls_tau"][best_epoch]
    return application_score

def get_mapping_tau(model, NocDataset, map_test_dir, epoch, show): 
    num_dirs        = len(os.listdir(map_test_dir))

    tau_list      = []
    p_value_list  = []
    std_list      = []
    count         = 0

    for i in range(num_dirs): 
        dir = os.path.join(map_test_dir, f"{i}")
        map_dataset = NocDataset(dir)

        truth_list = []
        pred_list  = []

        for j in range(len(map_dataset)): 
            data = map_dataset[j]
            
            mask_dict = get_mask_dict_from_data(data)
            if len(mask_dict) > 0:
                output  = model(data.x_dict, data.edge_index_dict, mask_dict)
            else:
                output  = model(data.x_dict, data.edge_index_dict)

            true_max_latency, pred_max_latency = get_max_latency_hetero(data, output)
            truth_list.append(true_max_latency)
            pred_list.append(pred_max_latency)
        
        tau, p_val = kendalltau(truth_list, pred_list)
        # p_val = 0
        # tau = adjusted_kendalls_tau(truth_list, pred_list, t_x=0, t_y=4)
        tau_list.append(tau)
        p_value_list.append(p_val)

        max_truth = max(truth_list) 
        min_truth = min(truth_list)
        std_truth = np.std(truth_list)
        std_list.append(std_truth)
        range_truth = max_truth - min_truth
        if show:
            print(f"{count}. Tau: {round(tau, 2)}, \tp_val: {round(p_val, 2)}, \trange: {round(range_truth, 2)}, \tStd: {round(std_truth, 2)}")
        count += 1

    average_tau     = round(sum(tau_list)/len(tau_list), 2)
    average_p_val   = round(sum(p_value_list)/len(p_value_list), 2)
    # std_tau         = round(np.std(tau_list), 2)

    print(f"Epoch: {epoch}\tAverage tau = {average_tau}\tAverage p_val = {average_p_val}")

    return average_tau, average_p_val


if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    parser.add_argument("--find", action="store_true", help="Find the epoch with the best tau")
    args = parser.parse_args()

    model_spec = importlib.util.spec_from_file_location("model", os.path.join(args.model_path, "model.py"))
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    dataset_spec = importlib.util.spec_from_file_location("dataset", os.path.join(args.model_path, "dataset.py"))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_module)

    # GNNHetero = model_module.GNNHetero
    HeteroGNN = model_module.HeteroGNN

    NocDataset = dataset_module.NocDataset

    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params              = yaml.safe_load(open(params_yaml_path))

    HIDDEN_CHANNELS     = params["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS      = params["NUM_MPN_LAYERS"]
    DATA_DIR            = params["DATA_DIR"]
    MESH_SIZE           = params["MESH_SIZE"]
    EMBEDDING_DIM       = params["EMBEDDING_DIM"]
    EPOCHS              = params["EPOCHS"]

    if "HETERO_AGGR" in params:
        HETERO_AGGR = params["HETERO_AGGR"]
        model = HeteroGNN( HIDDEN_CHANNELS, NUM_MPN_LAYERS, MESH_SIZE, EMBEDDING_DIM, HETERO_AGGR )
    else: 
        model = HeteroGNN( HIDDEN_CHANNELS, NUM_MPN_LAYERS, MESH_SIZE, EMBEDDING_DIM )
    
    map_test_dir = f"{DATA_DIR}/map_test"

    dataset = NocDataset(os.path.join(map_test_dir, "0"))
    data = dataset[0]

    mask_dict = get_mask_dict_from_data(data)
    if len(mask_dict) > 0:
        model(data.x_dict, data.edge_index_dict, mask_dict)
    else:
        model(data.x_dict, data.edge_index_dict)
    print(f"Model initialized")

    print(f"Using HeteroGNN")
    parameter_count = print_parameter_count(model)

    weight_paths = []
    model_path = os.path.join(args.model_path, "models")
    if not args.find: 
        show_tau    = True
        weight_path = get_weights_from_directory( model_path, args.epoch )
        weight_paths.append(weight_path)

    else: 
        show_tau     = False
        weight_paths = get_all_weights_from_directory(model_path)
    
    max_map_tau         = 0
    best_epoch          = 0
    its_p_val           = 0
    path_of_best_model  = ""

    # for debugging
    # application_score = get_application_score(weight_paths[0], total_epochs=EPOCHS, best_epoch=101)
    # exit()

    # for weight in weight_paths:
    #     print(f"Weight: {weight}")

    # exit()

    num_weights = len(weight_paths)
    print(f"Number of weights: {num_weights}")
    map_score_is_invalid = False

    for  i, weight_path in enumerate(weight_paths):

        print(f"\n[{i}]Loading model from {weight_path}")
        model.load_state_dict(torch.load(weight_path, weights_only=False))
        epoch = extract_epoch(weight_path)

        tau, p = get_mapping_tau(model, NocDataset, map_test_dir, epoch, show=show_tau)

        if np.isnan(tau):
            application_score, best_epoch = get_best_application_score(weight_path)
            max_map_tau = 0
            its_p_val = 0
            path_of_best_model = ""
            map_score_is_invalid = True
            break

        if tau > max_map_tau: 
            max_map_tau = tau
            best_epoch  = epoch
            its_p_val   = p
            path_of_best_model = weight_path

    if not map_score_is_invalid:
        application_score = get_application_score(path_of_best_model, EPOCHS, best_epoch)


    # Convert NumPy objects to standard Python types
    results_dict = {
        "best_epoch": int(best_epoch),  # Ensure it's saved as a string if required
        "application_score": float(application_score),
        "max_map_tau": float(max_map_tau),
        "parameter_count": int(parameter_count),    
        "p_val": float(its_p_val),
        "path_of_best_model": path_of_best_model,
    }

    # Define the YAML file path
    yaml_file_path = os.path.join(args.model_path, 'results.yaml')

    # Write the results to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(results_dict, yaml_file, default_flow_style=False)


    if not map_score_is_invalid:

        command = [
            "python3", "-m", "training.evaluate", 
            "--model_path", args.model_path,
            "--epoch", str(best_epoch), 
            "--with_network"
        ]
    
        subprocess.run(command)