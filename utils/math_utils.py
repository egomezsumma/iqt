import numpy as np
import sklearn.metrics as skm


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


#DEPRECADA
def mse_by_bval(m1,m2):
    means = []
    for i in range(0, m1.shape[3]):
        dif_pow2 = np.power((m2[:,:,:,i] - m1[:,:,:,i]), 2)
        means.append(dif_pow2.mean())
    return np.array(means, dtype=float)


def nmse(m1,m2, axis=None):
    """
    :param m1:
    :param m2:
    :param axis:
    :return: (sum (m1_i - m2_i / m2_i) ** 2 ) N
    """
    return (np.divide((m1-m2), m2)**2).mean(axis=axis)

def mse(m1,m2, axis=None):
    #print 'm1 min:max', m1.min(), m1.max()
    #print 'm2 min:max', m2.min(), m2.max()
    dif = m1 - m2
    #print 'dif min:max', dif.min(), dif.max()
    difcuad = dif**2
    #print 'difcuad min:max', difcuad.min(), difcuad.max()
    return (difcuad).mean(axis=axis)

def normalize(matrix):
    #norm = np.linalg.norm(matrix)
    return matrix/np.max(matrix)

## Calculo el coef de determinacion
from scipy import stats

def coef_det_4thdim(x, y):
    cdet = []
    size = x.shape[3]
    for i in range(0,size):
        _x = x[:,:,:,i].reshape(-1)
        _y = y[:,:,:,i].reshape(-1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(_x,_y)
        cdet.append(r_value**2)
    cdet=np.array(cdet)
    return cdet


def coef_det(x_orig, y, axis=None):
    """
    :param x_orig: valores originales
    :param y: valores estimados
    :param axis:
    :return:
        SS_res = sum (xi-yi)^2      --equiv-->  (mse(x,y))*N
        SS_tot = sum (xi -E(X))^2   --equiv-->  (var(x)*N)

        return 1-(SS_res/SS_tot)
    """
    _mse = mse(x_orig, y, axis)
    varx = np.var(x_orig, axis=axis)

    #cdet = np.ones(_mse.shape, dtype=float)- np.divide(_mse, varx)

    return 1 - np.divide(_mse, varx)

def coef_det_by_layer(x, y):
    return np.array([coef_det(x[:,:,:,i],y[:,:,:,i]) for i in range(0, x.shape[3])], dtype=float)

def coef_det_by_voxel(x, y):
    return coef_det(x, y, axis=3)


def coef_det_by_voxel_skl(x, y):
    cdet = np.zeros(x.shape[0:3], dtype=float)
    for i in range(0,x.shape[0]):
        for j in range(0, x.shape[1]):
            for k in range(0, x.shape[2]):
                _x = x[i, j, k, :].reshape(-1)
                _y = y[i, j, k, :].reshape(-1)
                cdet[i, j, k] =skm.r2_score(_x,_y)
    return cdet


def coef_det_by_layer_skl(x, y):
    cdet = []
    for i in range(0,x.shape[3]):
        _x = x[:,:,:,i].reshape(-1)
        _y = y[:,:,:,i].reshape(-1)
        cdet.append(skm.r2_score(_x,_y))
    return np.array(cdet, dtype=float)

"""
def coef_det2(x, y):
    cdet = np.zeros(x.shape[0:3], dtype=float)
    for i in range(0,x.shape[0]):
        for j in range(0, x.shape[1]):
            for k in range(0, x.shape[2]):
                _x = x[i, j, k, :].reshape(-1)
                _y = y[i, j, k, :].reshape(-1)
                _, _, r_value, _, _ = stats.linregress(_x,_y)
                cdet[i, j, k] =r_value**2
    return cdet
"""
