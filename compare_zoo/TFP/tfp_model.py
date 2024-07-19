import torch.nn as nn
import torch

class TFP(nn.Module):
    def __init__(self):
        super(TFP, self).__init__()
        pass

    def forward(self, spike,max_search_half_window = 20):
        mid = spike.shape[1] // 2
        spike = spike[:,mid-max_search_half_window:mid+max_search_half_window + 1]
        return torch.mean(spike,dim = 1,keepdim = True)