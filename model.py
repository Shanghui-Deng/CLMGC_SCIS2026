import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.functional import normalize
from utils.utils import build_affinity_matrix, sgc_precompute
from typing import Optional

EPS = 1e-12


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        xr = self.decoder(x)
        return xr


class ClusteringLayer(nn.Module):
    def __init__(self, class_num, hidden_dimension, alpha: float = 1.0, cluster_centers: Optional[torch.Tensor] = None):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param class_num: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(ClusteringLayer, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.class_num = class_num
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.class_num, self.hidden_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch):
        """
        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        # print('self.cluster_centers', self.cluster_centers.shape)
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, in_fea, out_fea):
        super(SGC, self).__init__()

        self.W = nn.Sequential(nn.Linear(in_fea, out_fea), nn.PReLU())

    def forward(self, x):
        x = self.W(x)
        return x


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num, device, temperature=.67):
        super(Network, self).__init__()

        self.encoders = []
        self.decoders = []
        self.cluster_projs = []
        self.view = view
        self.class_num = class_num
        self.device = device
        self.alpha = 1.0

        for v in range(view):
            encoder = Encoder(input_size[v], feature_dim)
            decoder = Decoder(input_size[v], feature_dim)
            self.encoders.append(encoder.to(device))
            self.decoders.append(decoder.to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)

        self.cross_projector = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            nn.ReLU()
        )

        self.cluster_proj = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.group_projector = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            Normalize(2),
        )



    def forward(self, xs):
        # zs = []
        xrs = []
        hs = []
        qs = []
        gs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.TransformerEncoderLayer(z)
            hs.append(h)
            xr = self.decoders[v](h)
            xrs.append(xr)
            v = self.cross_projector(z)
            q = self.cluster_proj(v)
            qs.append(q)
            g = self.group_projector(v)
            gs.append(g)

        return xrs, hs,qs,gs

    def forward_fusion(self,xs):
        xrs = []
        hs = []
        qs = []
        gs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.TransformerEncoderLayer(z)
            hs.append(h)
            xr = self.decoders[v](h)
            xrs.append(xr)
            v = self.cross_projector(z)
            q = self.cluster_proj(v)
            qs.append(q)
            g = self.group_projector(v)
            gs.append(g)



        return xrs, hs, qs, gs

    # def forward_fusion(self, xs):
    #     hs = []
    #     qs = []
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         z = self.encoders[v](x)
    #         zs.append(z)
    #         h = self.TransformerEncoderLayer(z)
    #         hs.append(h)
    #         q = self.cluster_proj(h)
    #         qs.append(q)
    #
    #     fusion_feautre = torch.cat(zs, dim=1)
    #     p = self.cluster_proj(fusion_feautre)
    #
    #     return hs, fusion_feautre, qs, p

    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.zeros(std.size()).normal_()
        if self.device:
            eps = eps.cuda()
        return mean + std * eps

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """

        # Sample from gumbel distribution
        unif = torch.rand(alpha.size())
        if self.device:
            unif = unif.cuda()
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        # Reparameterize to create gumbel softmax sample
        log_alpha = torch.log(alpha + EPS)
        logit = (log_alpha + gumbel) / self.temperature
        return F.softmax(logit, dim=1)
