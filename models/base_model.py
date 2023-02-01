from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from glob import glob


class GNC(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, stride, padding, dilation):
        super(GNC, self).__init__()
        self.linear_left = nn.Conv2d(input_channel, output_channel, kernel, stride, padding, dilation)
        self.linear_right = nn.Conv2d(input_channel, output_channel, kernel, stride, padding, dilation)

    def forward(self, x):
        return torch.mul(self.linear_left(x).tanh(), torch.sigmoid(self.linear_right(x)))


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample, nodes, length, dropout):
        super(Block, self).__init__()
        self.conv1 = GNC(in_channels, out_channels, (1, 3), stride=(1, stride), padding=(0, 1), dilation=1)
        self.norm = nn.LayerNorm((nodes, length), elementwise_affine=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm(x)
        # x = self.dropout(x)
        x = x + self.downsample(identity)
        return x


class STGC(nn.Module):
    def __init__(self, in_channels, nodes, length, momentum=0.1, alpha=1, padding=2, dropout=0.2):
        super(STGC, self).__init__()
        out_channels = in_channels//4
        # self.length = length
        self.padding = 0# length//2
        self.nodea = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1)))
        self.dropout = nn.Dropout(dropout)
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros([1, nodes, nodes, length]))
    
    def track(self, x):
        if self.training:
            with torch.no_grad():
                self.running_mean.copy_((1-self.momentum) * x.mean(dim=0) + self.momentum * self.running_mean)
        else:
            return self.running_mean#(x + self.running_mean)/2

    def forward(self, x):
        ########### 3d version
        # bs, 1, in_channels, node*length
        x = self.nodea(x)
        nodea = x.unsqueeze(1).flatten(start_dim=3)

        # nodea = self.nodea(x).unsqueeze(1).flatten(start_dim=3)

        # bs, node, in_channels, 1
        # nodeb = self.nodeb(x[..., -1:]).transpose(1, 2)
        nodeb = x[..., -1:].transpose(1, 2)
        bs, node, in_channels, _ = nodeb.shape
        sim = F.cosine_similarity(nodeb, nodea, dim=2).reshape(bs, node, node, -1)
        
        return sim


class TAST(torch.nn.Module):
    def __init__(self, nodes, in_channels, out_channels, improved=False, bias=True):
        super(TAST, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.e = nn.Conv2d(nodes, nodes, 1, groups=nodes)
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _, L = adj.size()
        if B == 1:
            adj = adj.repeat(x.shape[0], 1, 1, 1)
            B = adj.shape[0]
        
        ## separate version
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            # adj[..., -1] = 0
            adj[:, idx, idx, -1] = 1 if not self.improved else 2
        
        x = torch.einsum('bnml,beml->bnme', adj, x)

        # bs, node, node
        adj = F.relu(self.e(x).max(-1)[0].tanh())

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        adj = adj.squeeze(-1)
        adj = adj.sum(dim=-1, keepdim=True).clamp(min=1).reciprocal() * adj
        out = torch.einsum('bnme,bnm->bne', out, adj)

        if self.bias is not None:
            out = out + self.bias

        # bs, node, c
        return out


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels, stride, length, num_layers, nodes, has_gcn, dropout):
        super(Stage, self).__init__()
        self.has_gcn = has_gcn
        self.alpha = 0.1
        self.block = self.make_blocks(in_channels, out_channels, stride, length, num_layers, nodes, dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Conv2d(out_channels, middle_channels, 1)

        if has_gcn:
            self.gcn = TAST(nodes, out_channels, out_channels)
            self.norm = nn.LayerNorm((nodes, out_channels), elementwise_affine=True)
            self.edge = STGC(out_channels, nodes, length, dropout=dropout)
            self.out = nn.Sequential(
                nn.Conv2d(out_channels, middle_channels, 1))

    def make_blocks(self, in_channels, out_channels, stride, length, num_layers, nodes, dropout):
        blocks = []
        for i in range(num_layers):
            if i == 0:
                downsample = nn.Conv2d(in_channels, out_channels, 1, stride=(1,stride), padding=0) if stride > 1 or in_channels != out_channels else nn.Identity()
                blocks.append(
                    Block(in_channels, out_channels, stride, downsample, nodes, length, dropout))
            else:
                blocks.append(Block(out_channels, out_channels, 1, nn.Identity(), nodes, length, dropout))
        return nn.ModuleList(blocks)

    def forward(self, x, edge=None, node=None, index=None):
        for idx, b in enumerate(self.block):
            x = b(x)

        if self.has_gcn:
            if not torch.is_tensor(edge):
                edge = self.edge(x)

            x_gcn = self.gcn(x, edge)
            x_gcn = self.norm(x_gcn)

            x_gcn = x_gcn.transpose(1, 2).unsqueeze(-1)
            x_gcn = self.dropout(x_gcn)
            o = self.out(x_gcn)

            x = x + x_gcn#* self.attn(x_gcn).sigmoid()
            # print(o.shape)
            return x, o, edge
        o = self.out(x).mean(-1, keepdim=True)
        # print(o.shape)
        return x, o, None


class WSTGCN(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, step_type, stats, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(WSTGCN, self).__init__()
        self.stats = stats
        self.units = units
        self.iter = 1
        self.dropout_rate = 0.
        channels = [1, 32, 48, 56, 64]

        channels = [1] + [i * 1 for i in channels[1:]]
        # 2qq
        num_layers = [1, 2, 2, 2]

        strides = [1, 2, 2, 2]
        lengths = [1]+[math.ceil(time_step/np.prod(strides[:i+1])) for i in range(len(strides))]

        middle = channels[-1]
        self.dropout = nn.Dropout(0.)
        self.stages = nn.ModuleList([
            Stage(channels[idx], out_channel, middle, stride, length, num_layer, self.units, gcn_on, self.dropout_rate) 
                for idx, (out_channel, num_layer, stride, length, gcn_on) in enumerate(zip(channels[1:], num_layers, strides, lengths[1:], [False, False, False, True]))])
        self.horizon = horizon
        self.output = nn.Sequential(
            nn.Conv2d(middle, middle, 1),
            nn.ReLU(True),
            nn.Conv2d(middle, self.horizon if step_type == 'multi' else 1, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(1)
        r = []

        x, o, edge = self.stages[0](x)
        if torch.is_tensor(o):
            r.append(o)
        

        for i in range(1, len(self.stages)):

            x = self.dropout(x)
            x, o, _ = self.stages[i](x)
            if torch.is_tensor(o):
                r.append(o)
        r = sum(r)

        return self.output(r).squeeze(-1)
