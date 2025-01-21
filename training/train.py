import os 
import yaml
import random   
import argparse
import numpy as np

from tqdm        import tqdm
from scipy.stats import kendalltau

import torch
import torch.nn    as nn
import torch.optim as optim
from torch_scatter import scatter_max

from torch_geometric.data           import Data
from training.model_without_network import MPN, MPNHetero
from training.dataset               import load_data
from training.utils                 import ( does_path_exist, 
                                             does_model_dir_exit, 
                                             copy_file, 
                                             plot_and_save_loss, 
                                             print_parameter_count, 
                                             initialize_model, 
                                             get_mask_dict_from_data, 
                                             get_metadata )

def get_true_pred_hetero(data, output):
    """
    Given a HeteroData object and the model output,
    returns the true and predicted of each node type.  
    """
    true_task, pred_task, true_exit, pred_exit = None, None, None, None

    if 'task' in output and data['task'].y.numel() > 0:
        true_task = data['task'].y
        pred_task = output['task']
    
    if 'exit' in output and data['exit'].y.numel() > 0:
        true_exit = data['exit'].y
        pred_exit = output['exit']

    return true_task, pred_task, true_exit, pred_exit


def get_max_latency_hetero(data, output):
    """
    Note: NOT BATCHED
          Use torch_scatter.scatter_max for batched data
    Given a HeteroData object and the model output, 
    returns the maximum latency of the true and predicted values 
    """
    true_task, pred_task, true_exit, pred_exit = get_true_pred_hetero(data, output)

    true_max_task = np.nan
    pred_max_task = np.nan
    true_max_exit = np.nan
    pred_max_exit = np.nan

    if true_task is not None and pred_task is not None:
        argmax_task = torch.argmax(true_task[:, 1])  # Assuming latency is in column 1
        true_max_task = true_task[argmax_task, 1].item()
        pred_max_task = pred_task[argmax_task, 1].item()

    if true_exit is not None and pred_exit is not None:
        argmax_exit = torch.argmax(true_exit[:, 1])  # Assuming latency is in column 1
        true_max_exit = true_exit[argmax_exit, 1].item()
        pred_max_exit = pred_exit[argmax_exit, 1].item()

    true_max_latency = np.nanmax([true_max_task, true_max_exit])
    pred_max_latency = np.nanmax([pred_max_task, pred_max_exit])

    return true_max_latency, pred_max_latency


def process_batch(data, model, loss_fn, device):
    """
    Computes the loss for a batch of data
    """
    data    = data.to(device)

    if isinstance(data, Data):
        output  = model(data.x, data.edge_index)
        target_task = data.y.to(device)
        pred_task   = output
        loss        = loss_fn(pred_task, target_task)
        return loss
    
    mask_dict = get_mask_dict_from_data(data)
    if len(mask_dict) > 0:
        output = model(data.x_dict, data.edge_index_dict, mask_dict)
    else: 
        output = model(data.x_dict, data.edge_index_dict)

    true_task, pred_task, true_exit, pred_exit = get_true_pred_hetero(data, output)

    loss_task = loss_fn(pred_task, true_task) if true_task is not None else 0
    loss_exit = loss_fn(pred_exit, true_exit) if true_exit is not None else 0
    total_loss = loss_task + loss_exit

    # batch = data['task'].batch
    # _, max_indices = scatter_max(true_task[:, 1], batch)
    # total_loss = loss_fn(output['task'][max_indices, 1], true_task[max_indices, 1])

    return total_loss


def train_fn(train_loader, model, optimizer, loss_fn, device):
    loop        = tqdm(train_loader, leave=True)
    mean_loss   = []

    model.to(device)

    for batch_idx, data in enumerate(loop):

        loss = process_batch(data, model, loss_fn, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

        train_loss = sum(mean_loss) / len(mean_loss)

    return train_loss


def validation_fn(valid_loader, model, loss_fn, device):
    mean_loss = []

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            loss = process_batch(data, model, loss_fn, device)
            mean_loss.append(loss.item())

    validation_set_loss = sum(mean_loss) / len(mean_loss)

    return validation_set_loss


def test_fn(test_loader, model):
    ground_truth_latency_list   = []
    predicted_latency_list      = []

    model.to('cpu')
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data    = data.to('cpu')

            if isinstance(data, Data):
                output  = model(data.x, data.edge_index)
            else: 
                mask_dict = get_mask_dict_from_data(data)
                if len(mask_dict) > 0:
                    output  = model(data.x_dict, data.edge_index_dict, mask_dict)
                else:
                    output  = model(data.x_dict, data.edge_index_dict)
                
            latency_truth, latency_pred = get_max_latency_hetero(data, output)

            ground_truth_latency_list.append(latency_truth)
            predicted_latency_list.append(latency_pred)

    tau, p_value = kendalltau(ground_truth_latency_list, predicted_latency_list)

    return tau, p_value

def save_model(model, epoch, results_dir, test_metric, suffix=""):
    test_metric_int = int( round(test_metric, 3) * 100 )

    model_filename = f"{results_dir}/models/LatNet_{test_metric_int}_{epoch+1}_{suffix}.pth"
    torch.save(model.state_dict(), model_filename)

def train_and_validate(epochs, train_loader, valid_loader, test_loader, model, optimizer, loss_fn, device, save_path, save_threshold):

    train_loss_list, valid_loss_list, test_metric_list, saved_test_metric = [], [], [], []
    best_metric = 0

    for epoch in range(epochs):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, device=device)
        valid_loss = validation_fn(valid_loader, model, loss_fn, device=device)
        test_metric, pvalue = test_fn(test_loader, model)

        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {valid_loss}, Kendall's Tau: {test_metric}, P-Value: {round(pvalue, 5)}")
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        test_metric_list.append(test_metric)
        plot_and_save_loss(train_loss_list, valid_loss_list, test_metric_list, save_path)

        rounded_metric = round(test_metric, 3)
        if test_metric > save_threshold:
            if test_metric > best_metric and rounded_metric not in saved_test_metric:
                best_metric = test_metric
                saved_test_metric.append(rounded_metric)
                save_model(model, epoch, save_path, test_metric, suffix="best")
        if epoch % 10 == 0:
            save_model(model, epoch, save_path, test_metric, suffix="interval")

    save_model(model, epochs, save_path, test_metric_list[-1], suffix="last")


def main():

    parser = argparse.ArgumentParser(description="Train the GCN model")
    parser.add_argument( "name", type=str, help="Results will be saved in training/results/<name>")

    args = parser.parse_args()

    print(f"\nTraining Model without Network")

    model_path      = f"training/model_without_network.py"
    train_path      = f"training/train.py"
    params_path     = f"training/config/params_without_network.yaml"
    dataset_path    = f"training/dataset.py"

    TRAINING_PARAMS = yaml.safe_load(open(params_path))
    results_path    = TRAINING_PARAMS["RESULTS_DIR"]
    SAVE_PATH       = f"{results_path}/{args.name}"

    print(f"\nSaving Results to {SAVE_PATH}")

    does_path_exist(SAVE_PATH)
    does_model_dir_exit(SAVE_PATH)

    copy_file(model_path,   f"{SAVE_PATH}/model.py")
    copy_file(train_path,   f"{SAVE_PATH}/train.py")
    copy_file(params_path,  f"{SAVE_PATH}/params.yaml")
    copy_file(dataset_path, f"{SAVE_PATH}/dataset.py")

    # Seeds for Reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)

    # Training Parameters  
    NUM_MPN_LAYERS      = TRAINING_PARAMS["NUM_MPN_LAYERS"]
    HIDDEN_CHANNELS     = TRAINING_PARAMS["HIDDEN_CHANNELS"]
    LOSS_FN             = TRAINING_PARAMS["LOSS_FN"].lower()
    DEVICE              = TRAINING_PARAMS["DEVICE"]
    CONV_TYPE           = TRAINING_PARAMS["CONV_TYPE"]
    AGGR                = TRAINING_PARAMS["AGGR"]

    LEARNING_RATE       = TRAINING_PARAMS["LEARNING_RATE"]
    EPOCHS              = TRAINING_PARAMS["EPOCHS"]
    WEIGHT_DECAY        = TRAINING_PARAMS["WEIGHT_DECAY"]
    BATCH_SIZE          = TRAINING_PARAMS["BATCH_SIZE"]

    LOAD_MODEL          = TRAINING_PARAMS["LOAD_MODEL"]
    MODEL_PATH          = TRAINING_PARAMS["MODEL_PATH"]

    DATA_DIR            = TRAINING_PARAMS["DATA_DIR"]
    SAVE_THRESHOLD      = TRAINING_PARAMS["SAVE_THRESHOLD"]

    hetero_args = {
        "is_hetero"         : TRAINING_PARAMS["IS_HETERO"].strip().lower()       == "true", 
        "has_dependency"    : TRAINING_PARAMS["HAS_DEPENDENCY"].strip().lower()  == "true", 
        "has_exit"          : TRAINING_PARAMS["HAS_EXIT"].strip().lower()        == "true", 
        "has_scheduler"     : TRAINING_PARAMS["HAS_SCHEDULER"].strip().lower()   == "true"
    }
    print(f"Hetero parameters {hetero_args}")

    if DEVICE == "cuda":
        if torch.cuda.is_available():
            # Set Seeds for Cuda Reproducibility
            DEVICE = torch.device("cuda")
            torch.use_deterministic_algorithms(True)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else: 
            raise ValueError("CUDA is not available. Please set DEVICE='cpu' in params.yaml")

    elif DEVICE == "cpu":
        DEVICE = torch.device("cpu")

    else: 
        raise ValueError("DEVICE must be either 'cuda' or 'cpu'")

    print(f"\nTraining on {DEVICE}")

    train_data_dir  = f"{DATA_DIR}/train"
    test_data_dir   = f"{DATA_DIR}/test"

    train_loader, valid_loader  = load_data( train_data_dir, 
                                             batch_size          = BATCH_SIZE, 
                                             validation_split    = 0.1,
                                             use_noc_dataset     = False,
                                             **hetero_args )

    test_loader, _              = load_data( test_data_dir, 
                                             batch_size          = 1, 
                                             validation_split    = 0.0,
                                             use_noc_dataset     = False,
                                             **hetero_args )

    if hetero_args["is_hetero"]:
        model_type = "MPNHetero"
        metadata   = get_metadata( train_data_dir, **hetero_args )
        model      = MPNHetero( hidden_channels = HIDDEN_CHANNELS, 
                                num_mpn_layers  = NUM_MPN_LAYERS,
                                model_str       = CONV_TYPE,
                                metadata        = metadata ).to(DEVICE)

    else: 
        model_type  = "MPN"
        model       = MPN( hidden_channels  = HIDDEN_CHANNELS, 
                           output_channels  = 2, 
                           num_conv_layers  = NUM_MPN_LAYERS, 
                           model_str        = CONV_TYPE, 
                           aggr             = AGGR).to(DEVICE) 
    print(f"\nModel {model_type} with {CONV_TYPE}(aggr: {AGGR}) , loaded with {NUM_MPN_LAYERS} MPN Layers and {HIDDEN_CHANNELS} Hidden Channels")

    initialize_model(model, test_loader, DEVICE)
    print_parameter_count(model)

    if LOAD_MODEL:
        model_state_dict = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(model_state_dict)
        print(f"\nPre-Trained Wieghts Loaded\n")

    if LOSS_FN == "mse":
        loss_fn = nn.MSELoss().to(DEVICE)
    elif LOSS_FN == "mae":
        loss_fn = nn.L1Loss().to(DEVICE)
    elif LOSS_FN == "huber":
        loss_fn = nn.SmoothL1Loss().to(DEVICE)
    else: 
        raise ValueError("LOSS_FN must be either 'mse', 'mae', or 'huber'")

    print(f"Training with {LOSS_FN} Loss Function")

    optimizer   = optim.Adam( model.parameters(), 
                              lr=LEARNING_RATE, 
                              weight_decay=WEIGHT_DECAY )

    train_and_validate( epochs         = EPOCHS, 
                        train_loader   = train_loader, 
                        valid_loader   = valid_loader, 
                        test_loader    = test_loader, 
                        model          = model, 
                        optimizer      = optimizer, 
                        loss_fn        = loss_fn, 
                        device         = DEVICE, 
                        save_path      = SAVE_PATH, 
                        save_threshold = SAVE_THRESHOLD )

if __name__ == "__main__":
    main()
