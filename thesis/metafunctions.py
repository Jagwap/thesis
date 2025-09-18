from methods import embed_column, find_clusters
import numpy as np

def update_lr(batch, epoch,optimizer,range_coef=1.7, aim=5e-5, expected_lr=5e-5, goalr=5.0):
    av = batch["distances"]
    balanced = range_coef*((av/batch["mean"])-1) + 5e-5
    new_lr = max(1e-6, min(1e-4, balanced))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def embedding_distances(dataset):
        dataset = dataset.select(range(2000))
        emb = embed_column(dataset)
        centers = find_clusters(emb, n_clusters=5)
        distances = [min([np.linalg.norm(e - c) for c in centers]) for e in emb]
        dataset = dataset.add_column("distances", distances)
        dataset = dataset.add_column("mean",[np.mean(distances)]*len(distances))
        return dataset