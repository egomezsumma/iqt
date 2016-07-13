import numpy as np

def normalize_by_bval(m):
    """
        Tener cuidado de pasar matrices de floats
        O podria devolver muchos NaNs
        return m/||m||
    """
    res = np.ones(m.shape)
    for i in range(0, m.shape[3]):
        layer = m[:,:,:,i]
        layer_norm = np.linalg.norm(layer)
        val = layer/np.linalg.norm(layer)
        res[:,:,:,i] = val
    return res


def mse_by_bval(m1,m2):
    dif_pow2 = (m2 - m1)**2
    means = []
    for i in range(0, dif_pow2.shape[3]):
        means.append(dif_pow2[:,:,:,i].mean())
    return np.array(means, dtype=float)

def normalize(matrix):
    norm = np.linalg.norm(matrix)
    return matrix/norm