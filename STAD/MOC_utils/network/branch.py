from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch import nn


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.hm_2 = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm_2'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm_2[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

    def forward(self, input_chunk):
        input_chunk_1 = []
        input_chunk_2 = []
        for i in range(len(input_chunk)):
            input_chunk_1.append(input_chunk[i][0])
            input_chunk_2.append(input_chunk[i][1])

        output = {}
        output_wh = []
        for feature in input_chunk_1:
            output_wh.append(self.wh(feature))
        input_chunk_1 = torch.cat(input_chunk_1, dim=1)
        input_chunk_2 = torch.cat(input_chunk_2, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        output['hm'] = self.hm(input_chunk_1)
        output['hm_2'] = self.hm_2(input_chunk_2)
        output['mov'] = self.mov(input_chunk_1)
        output['wh'] = output_wh
        return output
