import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import global_mean_pool
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.distributions.multinomial import Multinomial
import math

class PHMLayer(nn.Module):
    def __init__(self, n, in_features, out_features):
        super(PHMLayer, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features

        self.bias = Parameter(torch.Tensor(out_features))

        self.a = torch.zeros((n, n, n))
        self.a = Parameter(torch.nn.init.xavier_uniform_(self.a))

        self.s = torch.zeros((n, self.out_features//n, self.in_features//n)) 
        self.s = Parameter(torch.nn.init.xavier_uniform_(self.s))

        self.weight = torch.zeros((self.out_features, self.in_features))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, a, b):
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def forward(self, input: Tensor) -> Tensor:
        self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
        input = input.type(dtype=self.weight.type())
        return F.linear(input, weight=self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None)
        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.a, a=math.sqrt(5))
        init.kaiming_uniform_(self.s, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

class PromptVQ(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 n_codebooks: int, 
                 n_samples: int,
                 temp = 1,
                 beta = 0.9):
        super(PromptVQ, self).__init__()
        self.n_codebooks = n_codebooks
        self.n_samples = n_samples
        self.beta = beta
        self._epsilon = 0.01
        self.temp = temp
        self.p_emb = nn.Parameter(torch.Tensor(1, in_channels))
        
        self.down_proj = PHMLayer(4, in_channels , hidden_channels) 
        self.up_proj = PHMLayer(4, hidden_channels, in_channels)

        self.codebook = nn.Parameter(torch.Tensor(n_codebooks, in_channels), requires_grad=False)  
        self.register_buffer('_ema_cluster_size', torch.ones(self.n_codebooks)/self.n_samples)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_emb)
        glorot(self.codebook)

    def add(self, x: Tensor, batch: Tensor, graph_emb: Tensor, training):

        device = x.device
        x_p = graph_emb

        h = self.down_proj(x_p)
        h = F.relu(h)
        p_c = self.up_proj(h)

        # Quantization
        p_q = []

        distances = (torch.sum(p_c**2, dim=1, keepdim=True) 
                        + torch.sum(self.codebook.data**2, dim=1)
                        - 2 * torch.matmul(p_c, self.codebook.data.t())) # num_graphs, num_codebooks
        
        multi = Multinomial(total_count=self.n_samples, logits=(-distances-1e-5)/self.temp)
        samples = multi.sample().to(device)


        p_q = torch.matmul(samples, self.codebook.data).view(p_c.shape) / self.n_samples
        prompt = p_c + (p_q - p_c).detach() + self.p_emb

        # consistency loss
        c_loss = torch.mean((p_q.detach() - p_c)**2)

        # EMA
        if training:
            self._ema_cluster_size = self._ema_cluster_size * self.beta + \
                                            (1 - self.beta) * \
                                            (torch.sum(samples, 0) / self.n_samples)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.n_codebooks * self._epsilon) * n)

            dw = torch.matmul(samples.t(), p_c) / self.n_samples
            normalized_ema_w = self.codebook.data * self.beta + (1 - self.beta) * (dw/self._ema_cluster_size.unsqueeze(1)) 
            self.codebook = nn.Parameter(normalized_ema_w, requires_grad=False) 

        return x + prompt, c_loss 
