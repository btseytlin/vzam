import faiss
import numpy as np


class FaissVideoSearcher:
    def __init__(self, vectors, labels, treshold=0.7, ncells=0.01):
        """
        N - total amount of video frames
        K - amount of features
        vectors: N x K frame descriptor vectors
        labels: N x 1 frame labels
        """
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        faiss.normalize_L2(vectors)
        self.vectors = np.ascontiguousarray(vectors)

        quantizer = faiss.IndexFlatL2(self.vectors.shape[1])  # the other index
        self.index = faiss.IndexIVFFlat(quantizer, self.vectors.shape[1],
                                        int(ncells * len(self.vectors)),
                                        faiss.METRIC_INNER_PRODUCT)
        self.index.train(self.vectors)
        self.index.add(self.vectors)
        self.labels = labels
        self.treshold = treshold

    def classify(self, vectors, threshold=None):
        """
        Majority vote
        """
        threshold = threshold or self.treshold
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        faiss.normalize_L2(vectors)
        vectors = np.ascontiguousarray(vectors)
        D, I = self.index.search(vectors, 1)
        min_indices, min_dists = I.flatten(), D.flatten()

        votes = self.labels[min_indices]
        miss_mask = min_dists < threshold
        votes[miss_mask] = 'miss'

        moc = max([(list(votes).count(chr), chr) for chr in set(votes)])
        moc = moc[1]
        return moc, votes, min_dists, min_indices
