import numpy as np

def normalize_by_bval(m):
    """
        Tener cuidado de pasar matrices de floats
        O podria devolver muchos NaNs
        return m/||m||
    """
    maxmax=-1
    res = np.ones(m.shape)
    for i in range(0, m.shape[3]):
        layer = m[:,:,:,i]
        layer_max = np.max(layer)
        maxmax = layer_max if layer_max > maxmax else maxmax
        val = layer/layer_max
        res[:,:,:,i] = val
    print 'maximo valor antes de normalizar :', maxmax
    return res


def mse_by_bval(m1,m2):
    means = []
    for i in range(0, m1.shape[3]):
        dif_pow2 = np.power((m2[:,:,:,i] - m1[:,:,:,i]), 2)
        means.append(dif_pow2.mean())
    return np.array(means, dtype=float)

def normalize(matrix):
    #norm = np.linalg.norm(matrix)
    return matrix/np.max(matrix)