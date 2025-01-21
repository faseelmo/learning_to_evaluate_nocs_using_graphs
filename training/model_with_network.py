import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch_geometric.nn     import ( GraphConv,
                                     HeteroConv )

from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.data import HeteroData

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_mpn_layers: int, mesh_size: int , embedding_dim: int , aggr_list: list[str]):  
        super().__init__()

        assert num_mpn_layers >= 2, "Number of layers should be at least 2."

        self._num_elems = mesh_size**2

        """For Unique Embeddings (non-isomorphic)"""
        self._pe_unique_embedding      = nn.Embedding(self._num_elems, embedding_dim)
        self._router_unique_embedding  = nn.Embedding(self._num_elems, embedding_dim)

        """For single embeddings (isomorphic)"""
        # self._pe_single_embedding      = nn.Embedding(1, embedding_dim)
        # self._router_single_embedding  = nn.Embedding(1, embedding_dim)

        """For Isomorphic Embeddings"""
        # self._corner_pe_embedding = nn.Embedding(1, embedding_dim)
        # self._border_pe_embedding = nn.Embedding(1, embedding_dim)
        # self._normal_pe_embedding = nn.Embedding(1, embedding_dim)

        # self._corner_router_embedding = nn.Embedding(1, embedding_dim)
        # self._border_router_embedding = nn.Embedding(1, embedding_dim)
        # self._normal_router_embedding = nn.Embedding(1, embedding_dim)

        self._convs     = nn.ModuleList()
        self._conv_aggr = MultiAggregation(["max"], mode='sum') 
        self._hetero_conv_aggr = aggr_list

        for _ in range(num_mpn_layers-1):
            intermediate_convs = self._get_hetero_conv(-1, hidden_channels)
            self._convs.extend(intermediate_convs)

        self._final_conv = self._get_hetero_conv(-1, 2)[0]


    def _get_hetero_conv(self, in_channels, out_channels): 
        
        conv_list = []

        for aggr in self._hetero_conv_aggr:

            conv = HeteroConv({
                ("task", "generates_for", "task"): GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
                ("task", "requires", "task"):      GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
                ("task", "mapped_to", "pe"):       GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("pe", "rev_mapped_to", "task"):   GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("router", "link", "router"):      GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("router", "interface", "pe"):     GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("pe", "rev_interface", "router"): GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            }, aggr=aggr)

            conv_list.append(conv)

        return conv_list

    def forward(self, x_dict, edge_index_dict, mask_dict) -> HeteroData:

        batch_size = x_dict['pe'].size(0) // self._num_elems

        """For unique embeddings (non-isomorphic)"""
        x_dict['pe'] = self._pe_unique_embedding.weight.repeat(batch_size, 1)
        x_dict['router'] = self._router_unique_embedding.weight.repeat(batch_size, 1)

        """For single embeddings (isomorphic)"""
        # x_dict['pe'] = self._pe_embedding.weight.repeat(self._num_elems, 1).repeat(batch_size, 1)
        # x_dict['router'] = self._router_embedding.weight.repeat(self._num_elems, 1).repeat(batch_size, 1)

        """For isomorphic embeddings"""
        # corner_router_mask = mask_dict['corner_router_mask'].bool()
        # border_router_mask = mask_dict['border_router_mask'].bool()
        # normal_router_mask = mask_dict['normal_router_mask'].bool()

        # corner_pe_mask = mask_dict['corner_pe_mask'].bool()
        # border_pe_mask = mask_dict['border_pe_mask'].bool()
        # normal_pe_mask = mask_dict['normal_pe_mask'].bool()

        # x_dict['router'] = torch.where(
        #         corner_router_mask.unsqueeze(-1),  
        #         self._corner_router_embedding.weight,  
        #         x_dict['router'])

        # x_dict['router'] = torch.where(
        #         border_router_mask.unsqueeze(-1),  
        #         self._border_router_embedding.weight,  
        #         x_dict['router'])

        # x_dict['router'] = torch.where(
        #         normal_router_mask.unsqueeze(-1),  
        #         self._normal_router_embedding.weight,  
        #         x_dict['router']   )

        # x_dict['pe'] = torch.where(
        #         corner_pe_mask.unsqueeze(-1),  
        #         self._corner_pe_embedding.weight,  
        #         x_dict['pe'])
        
        # x_dict['pe'] = torch.where(
        #         border_pe_mask.unsqueeze(-1),  
        #         self._border_pe_embedding.weight,  
        #         x_dict['pe'])
        
        # x_dict['pe'] = torch.where(
        #         normal_pe_mask.unsqueeze(-1),  
        #         self._normal_pe_embedding.weight,  
        #         x_dict['pe'])


        for conv in self._convs:

            x_dict = conv(x_dict, edge_index_dict)
            
            for key, x in x_dict.items():
                x_dict[key] = x.relu()

        x_dict = self._final_conv(x_dict, edge_index_dict)

        return x_dict


if __name__ == "__main__":

    """
    Usage: python3 -m training.model True False False False
    Conditions to Test: 
        1. Homogenous GNN Model
            python3 -m training.model 0 0 0 0 

        2. Heterogenous GNN Model
            python3 -m training.model 1 0 0 0 
            python3 -m training.model 1 0 1 0 (w/ wait time)
            python3 -m training.model 1 0 1 1 (w/ scheduler node and wait time)

        3. Heterogenous Pooling GNN Model
            [Works only for dataloader and not directly from CustomDataset. Issue with Batching]
            python3 -m training.model 1 1 0 0 
            python3 -m training.model 1 1 1 0
    """

    from training.dataset   import load_data
    from training.utils     import get_mask_dict_from_data

    IDX             = 10
    BATCH_SIZE      = 2
    HIDDEN_CHANNELS = 2
    MESH_SIZE       = 4

    torch.manual_seed(0)

    dataloader, _ = load_data( "data/training_data/with_network_4x4/test",
                                batch_size      = BATCH_SIZE,
                                use_noc_dataset = True )

    data        = next(iter(dataloader))
    print(f"Data shape is {data.x_dict['task'].shape}, {data.x_dict['pe'].shape}, {data.x_dict['router'].shape}")

    model       = HeteroGNN(HIDDEN_CHANNELS, num_mpn_layers=5, mesh_size=MESH_SIZE, embedding_dim=4, aggr_list=["sum"])
    
    mask_dict = get_mask_dict_from_data(data)
    print(f"Mask dict is \n{mask_dict}")

    output = model(data.x_dict, data.edge_index_dict, mask_dict)

    from training.utils import print_parameter_count
    print_parameter_count(model)





