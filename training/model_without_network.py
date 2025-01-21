
import importlib 

import torch
import torch.nn as nn

from torch_geometric.nn import to_hetero

class MPN(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels, model_str, num_conv_layers=3, aggr="sum"):
        super().__init__()
        torch.manual_seed(0)

        assert num_conv_layers >= 2, "Number of convolution layers should be at least 2."

        if model_str.lower() == "gcn":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "GCNConv")

        elif model_str.lower() == "graphconv":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "GraphConv")

        elif model_str.lower() == "gat":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "GATv2Conv")

        elif model_str.lower() == "sage":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "SAGEConv")

        else: 
            raise ValueError(f"Model {model_str} not found.")

        self.conv_list = nn.ModuleList()
        self.conv_list.append(CONV(-1, hidden_channels, aggr=aggr))

        for _ in range(num_conv_layers - 2):
            self.conv_list.append(CONV(hidden_channels, hidden_channels, aggr=aggr))

        self.conv_list.append(CONV(hidden_channels, output_channels, aggr=aggr))

    def forward(self, x, edge_index):
        for conv in self.conv_list[:-1]:
            x = conv(x, edge_index).relu()

        x = self.conv_list[-1](x, edge_index)
        
        return x

class MPNHetero(torch.nn.Module):
    def __init__(self, hidden_channels, num_mpn_layers, model_str, metadata):
        super().__init__()

        num_output_features     = 2
        mpn                     = MPN( hidden_channels  = hidden_channels, 
                                       output_channels  = num_output_features, 
                                       model_str        = model_str, 
                                       num_conv_layers  = num_mpn_layers )

        self.mpn_hetero         = to_hetero( module=mpn, 
                                             metadata=metadata, 
                                             aggr="sum", 
                                             debug=False)

    def forward(self, x_dict, edge_index_dict):
        out_dict        = self.mpn_hetero(x_dict, edge_index_dict)
        return out_dict



if __name__ == "__main__": 

    from training.utils import get_metadata
    from training.dataset import CustomDataset

    dataset_path = "data/training_data/without_network/test"
    num_layers   = 5

    hetero_dict = {
        "is_hetero"         : True, 
        "has_dependency"    : True,
        "has_exit"          : True,
        "has_scheduler"     : True   
    }

    dataset         = CustomDataset( dataset_path, **hetero_dict )
    data            = dataset[0]

    if hetero_dict["is_hetero"]: 
        print( f"Data is \n{data}" )
        metadata    = get_metadata( dataset_path, **hetero_dict )
        print(f"Metadata is \n{metadata}")
        model       = MPNHetero( hidden_channels=64, num_mpn_layers=num_layers, model_str="graphconv", metadata=metadata )
        output = model( data.x_dict, data.edge_index_dict )

        is_task_empty = data['task'].y.numel() == 0
        is_exit_empty = data['exit'].y.numel() == 0    

        print(f"Task is empty {is_task_empty}")
        print(f"Exit is empty {is_exit_empty}")


    else: 
        model = MPN(hidden_channels=64, output_channels=2, model_str="graphconv", num_conv_layers=num_layers)
        model(data.x, data.edge_index)

    # print(f"Model is \n{model}") 

