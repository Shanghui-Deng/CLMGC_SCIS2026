import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

    def cluster_orient(self,u1,u2,temperature=0.5):
        """
           cluster-level contrastive learning
           """
        # u1 = torch.mm(q1.T, z1)
        # u1 = F.normalize(u1, p=2, dim=1)
        #
        # u2 = torch.mm(q2.T, z2)
        # u2 = F.normalize(u2, p=2, dim=1)

        sim_cross1 = torch.exp(torch.mm(u1, u2.T) / temperature)
        sim_cross2 = torch.exp(torch.mm(u2, u1.T) / temperature)

        sim_same1 = torch.exp(torch.mm(u1, u1.T) / temperature)
        sim_same2 = torch.exp(torch.mm(u2, u2.T) / temperature)

        loss = 0
        loss += -torch.log(torch.diagonal(sim_cross1) / torch.sum(sim_cross1, dim=1)
                           ) + (torch.sum(sim_same1, dim=1) - torch.diagonal(sim_same1))
        loss += -torch.log(torch.diagonal(sim_cross2) / torch.sum(sim_cross2, dim=0)
                           ) + (torch.sum(sim_same1, dim=1) - torch.diagonal(sim_same1))

        return loss.mean()

    def cluster_feature(self,z1,z2,centers,label,cluster_num,temperature=0.5):
        loss = 0
        h1 = F.normalize(z1, p=2, dim=1)
        h2 = F.normalize(z2, p=2, dim=1)
        indicator = torch.ones(h1.size(0), cluster_num).to(h1.device)
        for i in range(self.class_num):
            indicator[i, label[i]] = 0
        sim_positive = torch.exp(torch.mm(h1, h2.T) / temperature)
        sim_negative = torch.mul(
            torch.exp(torch.mm(h1, centers.T) / temperature), indicator)

        loss = - torch.log(torch.diagonal(sim_positive) /
                           (torch.diagonal(sim_positive) + torch.sum(sim_negative, dim=1)))
        return loss.mean()


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    eps = 1e-8
    if input_as_probabilities:
        x_ = torch.clamp(x, min=eps)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0

    def forward(self, anchors, neighbors, cal_entropy=False):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        if cal_entropy:
            entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)
        else:
            entropy_loss = 0

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        # make scan loss positive
        # total_loss = consistency_loss + math.log(anchors_prob.size(1)) - entropy_loss

        return total_loss
