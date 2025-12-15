import torch
from torch import nn
from model import Network
from metric import valid
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import warnings
from sklearn.cluster import KMeans


from utils.utils import neraest_labels
import time
from scipy.optimize import linear_sum_assignment
from clusteringPerformance import clusteringMetrics
import contrastive_loss
from spectral_clustering import KMeans as Kmeans
from spectral_clustering import spectral_clustering

st = time.time()

torch.set_num_threads(4)
# MNIST-USPS
# BDGP
# LableMe
# Fashion
Dataname = 'DHA'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument('--neighbor_num', default=5, type=int)
parser.add_argument('--feature_dim', default=10, type=int)
parser.add_argument('--gcn_dim', default=128, type=int)
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--lambda1', default=1.0, type=float)
parser.add_argument('--lambda2', default=1.0, type=float)
parser.add_argument('--eta', default=1.0, type=float)
parser.add_argument('--neg_size', default=128, type=int)
parser.add_argument('--fine_epochs', default=50, type=int)
parser.add_argument('--instance_temperature', default=0.5, type=float)
parser.add_argument('--cluster_temperature', default=1.0, type=float)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore')


if args.dataset == 'COIL20':
    args.con_epochs = 200
    seed = 8
    args.learning_rate = 0.00001
if args.dataset =='MSRCv1':
    args.con_epochs = 600
    args.batch_size = 210
    seed = 10
    args.learning_rate = 0.00005
    # args.learning_rate = 0.00005
if args.dataset =='DHA':
    args.con_epochs = 300
    args.batch_size = 256
    seed = 3304
    args.learning_rate = 0.00008
if args.dataset == 'nus-wide':
    args.con_epochs = 300
    args.batch_size = 256
    seed = 10
    args.learning_rate = 0.000005

if args.dataset == 'flickr':
    args.con_epochs = 400
    args.batch_size = 256
    seed = 10
    args.learning_rate = 0.000005

if args.dataset == 'ESP-Game':
    args.con_epochs = 500
    args.batch_size = 256
    seed = 10
    args.learning_rate = 0.000005

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
data_loader_test = torch.utils.data.DataLoader(dataset,batch_size=data_size,
    shuffle=True,
    drop_last=True,)

# 添加噪声函数
def add_noise(input, noise_factor=0.1):
    noise = torch.randn_like(input) * noise_factor
    return input + noise


def pretrain(epoch):
    tot_loss = 0.
    for batch_idx, (xs, labels, _) in enumerate(data_loader):

        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        xrs, hs, qs, gs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(F.mse_loss(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    return hs, labels,model.state_dict()


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


def model_train(epoch):
    total_loss = 0.
    model.train()
    for batch_idx, (xs, labels, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, hs,qs,gs = model(xs)
        loss_instance = 0.
        loss_cluster = 0.
        for v in range(view):
            for w in range(v+1, view):
                loss_instance += criterion_instance(hs[v],hs[w])
                loss_cluster += criterion_cluster(qs[v],qs[w])

        use_kmeans = True
        cluster_num = 400  # k, 50, 100, 200, 400
        iter_num = 5
        k_eigen = class_num
        cld_t = 0.2
        cluster_labels = []
        centroids = []
        if use_kmeans:
            for v in range(view):
                cl_label, centroid = Kmeans(gs[v], K=cluster_num, Niters=iter_num)
                cluster_labels.append(cl_label)
                centroids.append(centroid)
        else:
            for v in range(view):
                cl_label, centroid = spectral_clustering(gs[v], K=k_eigen, clusters=cluster_num, Niters=iter_num)
                cluster_labels.append(cl_label)
                centroids.append(centroid)
                # instance-group discriminative learning

        criterion_cld = nn.CrossEntropyLoss().cuda()
        CLD_loss = 0
        for v in range(view):
            for w in range(view):
                if v != w:
                    affnity = torch.mm(gs[v], centroids[w].t())
                    CLD_loss = CLD_loss + criterion_cld(affnity.div_(cld_t), cluster_labels[v])

        CLD_loss = CLD_loss / view

        cross_loss = 0
        criterion_cross = nn.CrossEntropyLoss().cuda()
        loss_cf = []
        for v in range(view):
            for w in range(v, view):
                loss_cluster_feature = criterion_cluster.cluster_feature(hs[v], gs[w], centroid, cl_label,cluster_num)
                loss_cf.append(loss_cluster_feature.cpu().detach().numpy())

        # cross_loss = cross_loss / view
        loss_cf = sum(loss_cf)
        # loss_cc = sum(loss_cc)
        loss = loss_cluster  + CLD_loss + loss_instance + loss_cf
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss / len(data_loader)))
    return hs, labels



T = 1
for i in range(T):
    print("ROUND:{}".format(i + 1))

    # Network train
    model = Network(view, dims, args.feature_dim, class_num, device)
    # print(model)
    model = model.to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, device).to(device)

    epoch = 1
    best_acc = 0
    best_result = 0,0,0,0,0,0,0
    while epoch <= args.con_epochs:
        fusion, labels = model_train(epoch)
        acc_k, nmi_k, ari_k, pur_k, fscore_k, precision_k, recall_k = valid(
            model, device, dataset, view, data_size, class_num,
            eval_h=False)
        if acc_k >= best_result[0]:
            best_result = [acc_k, nmi_k, ari_k, pur_k, fscore_k, precision_k, recall_k]
        epoch += 1

    print('best result: ACC={:.4f}'.format(best_result[0]), 'NMI={:.4f}'.format(best_result[1]), 'ARI={:.4f}'.format(best_result[2]),
          'Pur={:.4f}'.format(best_result[3]), 'fscore={:.4f}'.format(best_result[4]), 'precision={:.4f}'.format(best_result[5]), 'recall={:.4f}'.format(best_result[6])
          )
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print('dataset:', args.dataset)
