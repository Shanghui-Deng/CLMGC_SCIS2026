from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import numpy as np
import torch
from clusteringPerformance import clusteringMetrics


def inference(loader, model, device, view,class_num, data_size):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high-level features
    Zs: low-level features
    """

    soft_vector = []
    pred_vectors = []
    Hs = []
    Zs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
        Zs.append([])
    labels_vector = []
    fusion_vector = []
    model.eval()
    with torch.no_grad():
        for step, (xs, y, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            xrs, hs, qs, gs = model(xs)
            q = sum(qs)/view

            for v in range(view):
                hs[v] = hs[v].detach()
                Hs[v].extend(hs[v].cpu().detach().numpy())

            soft_vector.extend(q.cpu().detach().numpy())
            labels_vector.extend(y.numpy())
    for v in range(view):
        Hs[v] = np.array(Hs[v])

    cat_feature = np.concatenate(Hs,axis=1)
    kmeans = KMeans(n_clusters=class_num,n_init=10)
    kmeans_vectors = kmeans.fit_predict(cat_feature)
    kmeans_vectors = kmeans_vectors.flatten()
    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector),axis=1)
    return total_pred, labels_vector,kmeans_vectors,cat_feature


def valid(model, device, dataset, view, data_size, class_num, eval_h=False):
    test_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    total_pred, labels_vector,kmeans_vectors,cat_feature = inference(test_loader, model, device, view,class_num, data_size)
    acc_k, nmi_k, ari_k, pur_k, fscore_k, precision_k, recall_k = clusteringMetrics(labels_vector, kmeans_vectors)
    return  acc_k, nmi_k, ari_k, pur_k, fscore_k, precision_k, recall_k
