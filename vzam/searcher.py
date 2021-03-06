import faiss
import numpy as np
import pandas as pd
import scipy


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

    def lookup_mv(self, vectors, dist_threshold=None):
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

    def lookup(self, vectors, conf_threshold=0.7):
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

    def __init__(self, vectors, labels, timestamps, dist_threshold=10, ncells=0.05):
        self.d = d = vectors.shape[1] * 8
        self.vectors = vectors.astype('uint8')
        self.labels = labels
        self.timestamps = timestamps
        self.dist_threshold = dist_threshold
        # self.quantizer = faiss.IndexBinaryFlat(self.d)
        # self.index = faiss.IndexIVFFlat(self.quantizer,
        #                                 self.d,
        #                                 int(ncells * len(self.vectors)))
        self.index = faiss.index_binary_factory(self.d, "BIVF32")

        self.index.train(self.vectors)
        self.index.add(self.vectors)

    def lookup_mv(self, vectors):
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

    def lookup(self, vectors, conf_threshold=0.7):
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

    def lookup_sequence(self, vectors):
        vectors = np.asarray(vectors).astype('uint8')
        D, I = self.index.search(vectors, 10)

        sequence_starts = I[0]
        candidates = []
        for i, start_index in enumerate(sequence_starts):
            dist = D[0][i]
            if dist > self.dist_threshold:
                continue
            candidate = {'start_index': start_index,
                         'dist': dist,
                         'label': self.labels[start_index],
                         'ts': self.timestamps.values[start_index],
                         }
            candidates.append(candidate)

        seq_candidates = []
        for candidate in candidates:
            # Extract sequence for that candidate
            index_seq = slice(candidate['start_index'] + 1,
                              candidate['start_index'] + len(vectors) - 1)
            label_sequence = self.labels[index_seq]
            if len(np.unique(label_sequence)) != 1:
                # print('impossible candidate', candidate)
                # print(label_sequence)
                continue

            source_vectors = self.vectors[index_seq]
            distances = []
            for qv, v in zip(vectors, source_vectors):
                distances.append(scipy.spatial.distance.hamming(qv, v))
            total_dist = (candidate['dist'] / len(vectors[0]) + sum(
                distances)) / len(vectors)
            seq_candidates.append({
                'start_index': candidate['start_index'],
                'ts': candidate['ts'],
                'label': candidate['label'],
                'dist': total_dist,
            })
        seq_candidates = sorted(seq_candidates, key=lambda x: x['dist'])
        return seq_candidates