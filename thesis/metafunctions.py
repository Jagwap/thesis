from thesis.methods import *
from thesis.embeddings import *
import numpy as np

def update_lr(batch, epoch,optimizer,range_coef=1.7, aim=5e-5, expected_lr=5e-5):
        av = batch["distances"]
        balanced = range_coef*((av/batch["mean"])-1) + expected_lr
        new_lr = max(1e-6, min(1e-4, balanced))
        for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

def embedding_distances(dataset, column = "context",size=2000, n_clusters=6):
        dataset = dataset.select(range(size))
        emb = EmbeddingModel.get_embeddings(dataset[column])
        print("done")
        centers = find_clusters(emb, n_clusters=n_clusters)
        distances = [min([np.linalg.norm(e - c) for c in centers]) for e in emb]
        dataset = dataset.add_column("distances", distances)
        dataset = dataset.add_column("mean",[np.mean(distances)]*len(distances))
        return dataset

def embedding_sampling(dataset, column = "context",size=2000, subset_size=1000, n_clusters=6):
        dataset = dataset.select(range(size))
        emb = EmbeddingModel.get_embeddings(dataset[column])
        print("done")
        centers = find_clusters(emb, n_clusters=n_clusters)
        distances = [min([np.linalg.norm(e - c) for c in centers]) for e in emb]
        dataset = dataset.add_column("distances", distances)
        dataset = dataset.sort("distances")
        return dataset.select(range(size-subset_size,size))

def equal_sampling(dataset, column = "context",size=2000, subset_size=1000, n_clusters=6):
        dataset = dataset.select(range(size))
        emb = EmbeddingModel.get_embeddings(dataset[column])
        print("done")
        centers = find_clusters(emb, n_clusters=n_clusters)
        distances = [min([np.linalg.norm(e - c) for c in centers]) for e in emb]
        dataset = dataset.add_column("distances", distances)
        dataset = dataset.sort("distances")
        return dataset.select(fps(dataset["distances"],subset_size))