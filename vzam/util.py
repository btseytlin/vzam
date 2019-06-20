from sklearn.preprocessing import normalize


def normalize_l1(arr):
    return normalize(arr.reshape(1, -1), 'l1').flatten()


