import numpy as np
from utils import img_utils
import mymapl.minmapl as mapl

#from mapmri import minmapl as mapl



def get_sample_dwi(index, numbers, loader_func, bval=None, bvalpos=None, scale=2):
    """
    Get the sample for the subjets numbers[index] and retur
    :param index: index on numbers of subject needed
    :param numbers: reference numbers for subject
    :param loader_func: funcion that load a especific subject
    :param bval: for single-shell data the only bval for buil the gtab
    :param bvalpos: positions in the data where are the volumes with b-value = bval
    :param scale: scaling for the downsampling
    :return: Hr, Lr, S0Hr, S0Lr, gtab
    """
    # Load Hcp subject
    img, gtab, idxs = loader_func(index, numbers, bval, bvalpos)
    # Downsample data
    lr, _ = img_utils.downsampling(img, scale)
    data = img.get_data()
    data_noS0 = data[:, :, :, idxs]
    s0_idxs = [i for i in xrange(max(idxs) + 1) if i not in idxs]
    return data_noS0, lr[:, :, :, idxs], data[:, :, :, s0_idxs], lr[:, :, :, s0_idxs], gtab


def get_sample_of_dwi(index, numbers, loader_func, bval=None, bvalpos=None, scale=2):
    hr, lr, S0hr, S0lr, gtab = get_sample_dwi(index, numbers, loader_func, bval=None, bvalpos=None, scale=2)
    hr = get_atenuation(hr, S0hr)
    lr = get_atenuation(lr, S0lr)
    del (S0hr, S0lr)
    return hr, lr, gtab


def get_sample_of_mapl(index, numbers, loader_func, bval=None, bvalpos=None, scale=2, multiply_S0=False):
    hr, lr, S0hr, S0lr, gtab = get_sample_dwi(index, numbers, loader_func, bval=None, bvalpos=None, scale=2)
    # Calculate MAPL  C_hr:(Nx,Ny,Nz,Nc) c_lr:(nx,ny,nz,nc)
    C_hr = mapl.getC(hr, gtab, radial_order=4)
    c_lr = mapl.getC(lr, gtab, radial_order=4)

    # Multiply by S0 to get the signal (and not the atenuation)
    if multiply_S0:
        C_hr = get_signal(C_hr, S0hr)
        c_lr = get_signal(c_lr, S0lr)

    # Clean-up
    del (hr)
    del (lr)
    del (S0hr)
    del (S0lr)
    return C_hr, c_lr, gtab


def get_sample_maker_of_map(numbers, loader_func, bval=None, bvalpos=None, scale=2):
    return lambda index: get_sample_of_mapl(index, numbers, loader_func, bval, bvalpos, scale)


def get_sample_maker_of_dwi(numbers, loader_func, bval=None, bvalpos=None, scale=2):
    return lambda index: get_sample_of_dwi(index, numbers, loader_func, bval, bvalpos, scale)


def buildT(sample_getter, n_samples):
    C, c, _ = sample_getter(0)
    X = img_utils.column_this(c)
    Y = img_utils.column_this(C)
    for i in range(1, n_samples):
        noised_hr, noised_lr, _ = sample_getter(i)
        X = img_utils.append_column(X, c)
        Y = img_utils.append_column(Y, C)
    return X, Y


def buildT_grouping_by(sample_getter, n_samples, values_needed=None):
    """
    Genera tantos conjuntos de entrenamiento como
    bvals distintos tenga el volumne
    """
    hr, lr, gtab = sample_getter(0)
    bs=None
    if values_needed is not None:
        bs = values_needed[0:hr.shape[3]]

    dicX = split_by(lr, gtab, vals_needed=bs)
    dicY = split_by(hr, gtab, vals_needed=bs)
    for i in range(1, n_samples):
        hr, lr, gtab = sample_getter(i)
        dicX = split_by(lr, gtab, dicX, vals_needed=bs)
        dicY = split_by(hr, gtab, dicY, vals_needed=bs)
    return dicX, dicY

"""
def buildT_grouping_by_c(sample_getter, n_samples):
    Genera tantos conjuntos de entrenamiento coef de MAPL

    C, c, _ = sample_getter(0)
    dicX = split_by_coef(c)
    dicY = split_by_coef(C)
    for i in range(1, n_samples):
        C, c, _ = sample_getter(i)
        dicX = split_by_coef(c, dicX)
        dicY = split_by_coef(C, dicY)
    return dicX, dicY
"""
def split_by(img, gtab, res=None, vals_needed=None):
    if vals_needed is None:
        return split_by_coef(img, res)
    else:
        return split_by_bval(img, gtab, vals_needed, res)

def split_by_bval(img, gtab, bvals_needed, res=None):
    """
    Dada una imagen separa la cuarta dimension segun su vbal
    Y por cada una hace un vector columna
    """
    if res is None:
        res = dict((b, None) for b in bvals_needed)

    for i in xrange(len(gtab.bvals)):
        b = gtab.bvals[i]
        if b not in bvals_needed:
            continue

        XorY = res[b]
        if XorY is None:
            res[b] = img_utils.column_this(img[:, :, :, i])
        else:
            res[b] = img_utils.append_column(XorY, img[:, :, :, i])
    return res

def split_by_coef(img, res=None):
    """
    Dada una imagen separa la cuarta dimension segun su coef de MAPL
    """
    Nc = img.shape[3]
    if res is None:
        res = dict((coef, None) for coef in xrange(Nc))

    for c in xrange(Nc):
        XorY = res[c]
        if XorY is None:
            res[c] = img_utils.column_this(img[:, :, :, c])
        else:
            res[c] = img_utils.append_column(XorY, img[:, :, :, c])
    return res


def get_atenuation(Sq, S0):
    _S0 = S0
    if len(S0.shape) > 3:
        _S0 = S0.mean(axis=3)
    for b in xrange(Sq.shape[3]):
        Sq[:, :, :, b] = np.divide(Sq[:, :, :, b], _S0)
    return Sq


def get_signal(Eq, S0):
    _S0 = S0
    if len(S0.shape) > 3:
        _S0 = S0.mean(axis=3)
    for b in xrange(Eq.shape[3]):
        Eq[:, :, :, b] = np.multiply(Eq[:, :, :, b], _S0)
    return Eq


    ## Example of use
    # buildT(get_sample_maker(numbers, scale), n_samples)