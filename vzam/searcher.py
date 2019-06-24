import faiss
import numpy as np
import pandas as pd


def clear_label_df(label_df):
    bad_indices = []

    current_query_pos = None
    earliest_query_ts = {}
    for index, row in label_df.iterrows():
        if not current_query_pos:
            current_query_pos = row.query_vec
        if row.query_vec > current_query_pos and row.ts < earliest_query_ts[
            current_query_pos]:
            bad_indices.append(index)
            continue

        if not earliest_query_ts.get(row.query_vec) or row.ts < earliest_query_ts[
            row.query_vec]:
            earliest_query_ts[row.query_vec] = row.ts

        if row.query_vec > current_query_pos + 1:
            current_query_pos = row.query_vec

    clean_label_df = label_df.drop(bad_indices)

    ts = clean_label_df.ts
    clean_label_df = clean_label_df[
        (ts.median() - 3 * ts.std() <= ts) & (ts <= ts.median() + 3 * ts.std())]
    clean_label_df = clean_label_df.groupby(['query_vec'])[
        'dist', 'label', 'ts'].min()
    return clean_label_df


class FaissVideoSearcher:
    def __init__(self, vectors, labels, timestamps, dist_threshold=0.7, ncells=0.01):
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
        self.timestamps = timestamps
        self.dist_threshold = dist_threshold

    def lookup(self, vectors, dist_threshold=None):
        """
        Majority vote
        """
        threshold = dist_threshold or self.dist_threshold
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

    def lookup_fun(self, vectors, conf_threshold=0.7):
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        faiss.normalize_L2(vectors)
        vectors = np.ascontiguousarray(vectors)
        D, I = self.index.search(vectors, 10)

        labels = self.labels[I]
        timestamps = self.timestamps.values[I]
        candidates = []
        for i in range(len(vectors)):
            for j in range(I.shape[1]):
                dist = D[i][j]
                if dist < self.dist_threshold:
                    continue
                candidate = {'query_vec': i,
                             'dist': D[i][j],
                             'label': labels[i][j],
                             'ts': timestamps[i][j],
                             }
                candidates.append(candidate)
        df = pd.DataFrame(candidates, columns=['query_vec', 'dist', 'label',
                                               'ts']).sort_values(
            by=['label', 'query_vec', 'ts'])
        potential_labels = df.label.unique()
        predictions = []
        for label in potential_labels:
            label_df = df[df.label == label]
            clean_label_df = clear_label_df(label_df)
            confidence = len(clean_label_df) / len(vectors)
            if confidence < conf_threshold:
                continue
            mean_dist = clean_label_df.dist.mean()
            predictions.append((label, confidence, mean_dist))
        predictions = sorted(predictions, key=lambda x: (-x[1], x[2]))
        return predictions


class FaissRhashVideoSearcher:

    def __init__(self, vectors, labels, timestamps, dist_threshold=10):
        self.d = d = vectors.shape[1]
        self.vectors = vectors.astype('uint8')
        self.labels = labels
        self.timestamps = timestamps
        self.dist_threshold = dist_threshold
        self.index = faiss.IndexBinaryFlat(self.vectors.shape[1] * 8)
        # self.index = faiss.IndexIVFFlat(self.quantizer,
        #                                 self.vectors.shape[1],
        #                                 int(ncells * len(self.vectors)))
        # self.index = faiss.index_binary_factory(d, "BIVF32")

        self.index.train(self.vectors)
        self.index.add(self.vectors)

    def lookup(self, vectors):
        vectors = np.asarray(vectors).astype('uint8')
        D, I = self.index.search(vectors, 1)
        min_indices, min_dists = I.flatten(), D.flatten()
        votes = self.labels[min_indices]
        timestamps = self.timestamps[min_indices]
        miss_mask = min_dists > self.dist_threshold
        votes[miss_mask] = 'miss'

        moc = max([(list(votes).count(chr), chr) for chr in set(votes)])
        moc = moc[1]
        return moc, votes, timestamps, min_dists, min_indices

    def lookup_fun(self, vectors, conf_threshold=0.7):
        vectors = np.asarray(vectors).astype('uint8')
        D, I = self.index.search(vectors, 10)

        labels = self.labels[I]
        timestamps = self.timestamps.values[I]
        candidates = []
        for i in range(len(vectors)):
            for j in range(I.shape[1]):
                dist = D[i][j]
                if dist > self.dist_threshold:
                    continue
                candidate = {'query_vec': i,
                             'dist': D[i][j],
                             'label': labels[i][j],
                             'ts': timestamps[i][j],
                             }
                candidates.append(candidate)
        df = pd.DataFrame(candidates, columns=['query_vec', 'dist', 'label',
                                               'ts']).sort_values(
            by=['label', 'query_vec', 'ts'])
        potential_labels = df.label.unique()
        predictions = []
        for label in potential_labels:
            label_df = df[df.label == label]
            clean_label_df = clear_label_df(label_df)
            confidence = len(clean_label_df) / len(vectors)
            if confidence < conf_threshold:
                continue
            mean_dist = clean_label_df.dist.mean()
            predictions.append((label, confidence, mean_dist))
        predictions = sorted(predictions, key=lambda x: (-x[1], x[2]))
        return predictions