import numpy as np
from utils import img_utils
import mymapl.minmapl as mapl

#from mapmri import minmapl as mapl



def get_sample_dwi(subject_number,i,j,k, loader_func, bval=None, bvalpos=None,bsize=-1, scale=2):
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
    print 'bsize=', bsize
    img, gtab, idxs = loader_func(subject_number,i,j,k, bval, bvalpos, bsize=bsize)
    # Downsample data
    lr, _ = img_utils.downsampling(img, scale)
    print 'scale=', scale, 'lr.shape=', lr.shape
    data = img.get_data()
    data_noS0 = data[:, :, :, idxs]
    s0_idxs = [i for i in xrange(max(idxs) + 1) if i not in idxs]
    return data_noS0, lr[:, :, :, idxs], data[:, :, :, s0_idxs], lr[:, :, :, s0_idxs], gtab


def get_sample_of_dwi(subject_number, i,j,k, loader_func, bval=None, bvalpos=None, bsize=-1, scale=2):
    hr, lr, S0hr, S0lr, gtab = get_sample_dwi(subject_number,i,j,k, loader_func, bval=None, bvalpos=None, bsize=bsize, scale=scale)
    hr = get_atenuation(hr, S0hr)
    lr = get_atenuation(lr, S0lr)
    del (S0hr, S0lr)
    return hr, lr, gtab


def get_sample_of_mapl(subject_number, i, j, k, loader_func, bval=None, bvalpos=None, bsize=-1, scale=2, multiply_S0=False):
    hr, lr, S0hr, S0lr, gtab = get_sample_dwi(subject_number, i,j,k, loader_func, bval=bval, bvalpos=bvalpos, bsize=bsize, scale=scale)
    print 'hr:', hr.shape, 'lr:', lr.shape,

    # Calculate MAPL  C_hr:(Nx,Ny,Nz,Nc) c_lr:(nx,ny,nz,nc)
    C_hr = mapl.getC(hr, gtab, radial_order=4)
    c_lr = mapl.getC(lr, gtab, radial_order=4)

    # Multiply by S0 to get the signal (and not the atenuation)
    if multiply_S0:
        C_hr = get_signal(C_hr, S0hr)
        c_lr = get_signal(c_lr, S0lr)

    # Clean-up
    #del (hr)
    #del (lr)
    del (S0hr)
    del (S0lr)
    return C_hr, c_lr, gtab, hr, lr


def get_sample_maker_of_map(loader_func, bval=None, bvalpos=None, bsize=-1):
    return lambda subject_num, i, j ,k, scale : get_sample_of_mapl(subject_num, i, j ,k , loader_func, bval, bvalpos, bsize=bsize, scale=scale)

def get_sample_maker_of_dwi(loader_func, bval=None, bvalpos=None, bsize=-1):
    return lambda subject_num, i, j ,k, scale : get_sample_of_dwi(subject_num, i, j ,k , loader_func, bval, bvalpos, bsize=bsize, scale=scale)


def buildT(sample_getter, n_samples):
    C, c, _ = sample_getter(0)
    X = img_utils.column_this(c)
    Y = img_utils.column_this(C)
    for i in range(1, n_samples):
        noised_hr, noised_lr, _ = sample_getter(i)
        X = img_utils.append_column(X, c)
        Y = img_utils.append_column(Y, C)
    return X, Y


def buildT_grouping_by(subjects,i, j, k, sample_getter, use_bvals=False,scale=2):
    """
    Genera tantos conjuntos de entrenamiento como
    bvals distintos tenga el volumne
    """
    if use_bvals :
        hr, lr, gtab = sample_getter(subjects[0], i, j, k, scale)
    else:
        hr, lr, gtab , _, _ = sample_getter(subjects[0], i, j, k, scale)

    dicX = split_by(lr, gtab, use_bvals=use_bvals)
    dicY = split_by(hr, gtab, use_bvals=use_bvals)
    for i in range(1, len(subjects)):
        subject = subjects[i]

        if use_bvals:
            hr, lr, gtab = sample_getter(subject, i, j, k, scale)
        else:
            hr, lr, gtab, _, _ = sample_getter(subjects[0], i, j, k, scale)

        dicX = split_by(lr, gtab, dicX, use_bvals=use_bvals)
        dicY = split_by(hr, gtab, dicY, use_bvals=use_bvals)
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
def split_by(img, gtab=None, res=None, use_bvals=False):
    if use_bvals :
        return split_by_bval(img, gtab, res=res)
    else:
        return split_by_coef(img, res)

def split_by_bval(img, gtab, bvals_needed=None, res=None):
    """
    Dada una imagen separa la cuarta dimension segun su vbal
    Y por cada una hace un vector columna
    """
    bvals_needed = gtab.bvals if bvals_needed is None else bvals_needed

    if res is None:
        res = dict((b, None) for b in bvals_needed)

    for i in xrange(len(gtab.bvals)):
        b = gtab.bvals[i]

        if bvals_needed is not None and b not in bvals_needed:
            continue

        if b not in res.keys():
            res[b] = None

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
    _S0[_S0==0.0] = 0.0001 # para que no explote la division por cero
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
