import numpy as np
import torch
import cv2
import torch.nn as nn
import torch

class TFI(nn.Module):
    def __init__(self):
        super(TFI, self).__init__()
        self.device = 'cuda'

    def forward(self, spike, max_search_half_window=20):
        bs = spike.shape[0]
        T = spike.shape[1]
        spike_h = spike.shape[2]
        spike_w = spike.shape[3]
        key_ts = T // 2
        formmer_index = torch.zeros([bs,spike_h, spike_w]).to(self.device)
        latter_index = torch.zeros([bs,spike_h, spike_w]).to(self.device)

        start_t = max(key_ts - max_search_half_window + 1, 1)
        end_t = min(key_ts + max_search_half_window, T)

        for ii in range(key_ts, start_t-1, -1):
            formmer_index += ii * spike[:,ii, :, :] * (1 - torch.sign(formmer_index).to(self.device))

        for ii in range(key_ts+1, end_t+1):
            latter_index += ii * spike[:,ii, :, :] * (1 - torch.sign(latter_index).to(self.device))

        interval = latter_index - formmer_index
        interval[interval == 0] = 2*max_search_half_window
        interval[latter_index == 0] = 2*max_search_half_window
        interval[formmer_index == 0] = 2*max_search_half_window
        interval = interval

        Image = 1 / interval
        return Image[:,None]