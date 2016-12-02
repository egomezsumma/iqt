import time
import numpy as np
import cvxpy as cvx
import load.hcp_img_loader as hcp
import iterators.DmriPatchIterator as d
from scipy.sparse import csr_matrix
import experimento1_funciones as e1f
import load.samples as samples
import sys
#import gc
import datetime
from utils.persistance.ResultManager import ResultManager
import warnings

#from threading import Thread, Lock
#from multiprocessing import Pool

#gc.enable()

#def uprint(*msg):
#    print msg
#    sys.stdout.flush()

t1, t2, t3, t4 = ' '*4,' '*8, ' '*12, ' '*16


IS_NEF = '/home/lgomez/' in sys.prefix

if IS_NEF:
    from utils.persistence_array import parray
else:
    from utils.persistance.persistence_array import parray


def mm(A, cast_int=True):
    if cast_int:
        return (int(A.min()), int(A.max()))
    else:
        return (A.min(), A.max())


# ## MapMri

import mymapl.minmapl as mapl

# # 3D Tv-Norm
# $ \sum_{ijk \in positions(I)} \left|\left| (I_{i,j,k}, I_{i,j,k}, I_{i,j,k}) - ( I_{i-1,j,k}, I_{i,j-1,k},  I_{i,j,k-1}) \right|\right| $

import optimization.tvnorm3d as tvn


# ## Problem definition
# 
# $ \min_{C^{hr}} \{ ||GC^{hr} - C^{lr}||^2  + ||C||_{1} \}$
# 
from utils import img_utils


# In[5]:
def define_problem_f1(c_lr, vhr, vlr, G, M, U, tau, gtab, scale,c_hr_initial=None, intercept=None, toprint=''):
    Nx, Ny, Nz = (12, 12, 12)#TODO: pasar
    Nb, Nc = M.shape

    ## LR volumes
    Clr = c_lr
    print '%%%%%%%%%',toprint, id(Clr)    

    ## MAPL params
    #cvxChr = cvx.Constant(C_hr.reshape(-1, order='F'))
    
    cvxChr = cvx.Variable(vhr*Nc, name='cvxChr')
    if c_hr_initial is None:
        ChrValueInitial = np.ones((vhr*Nc, 1), dtype='float32')
        for c in xrange(Nc): 
            c_offset_hr = c*vhr
            #import pdb; pdb.set_trace()
            # nose porque me hace un borde de 1 si no es par el original
            upsampling = img_utils.downsampling2(Clr[c].reshape((nx, ny, nz)), 0.5)[1:-1, 1:-1, 1:-1]
            print 'upsampling.shape', upsampling.shape
            # solo el centro
            upsampling = upsampling[4:6, 4:6, 4:6]
            #ChrValueInitial[c_offset_hr:c_offset_hr+vhr] = ChrValueInitial[c_offset_hr:c_offset_hr+vhr]*Clr[c].mean()
            print ChrValueInitial[c_offset_hr:c_offset_hr+vhr].shape, upsampling.reshape((Nx*Ny*Nz,1), order='F').shape
            ChrValueInitial[c_offset_hr:c_offset_hr+vhr] = upsampling.reshape((Nx*Ny*Nz,1), order='F')
        cvxChr.value = ChrValueInitial
    else:
        cvxChr.value = c_hr_initial.reshape(-1, order='F')

    ## Fidelity expression
    cvxG = G
    fidelity_list = []
    for c in xrange(Nc):
        c_offset_hr = c*vhr
        Chr_c = cvxChr[c_offset_hr:c_offset_hr+vhr]

        Gc = cvx.Constant(G[c])
        Clr_c = cvx.Constant(Clr[c])

        # Gc:(216:vlr, 1728:vhr) Chr_c:(1728:vhr, 1) Clr_c:(1, 216:vlr)
        if intercept is not None:
            cvxInt_c = cvx.Constant(intercept[c])
            # cvxInt_c:(vlr, 1)
            fid_b = cvx.sum_squares((Gc * Chr_c + cvxInt_c) - Clr_c)
            # fid_b = cvx.sum_squares((Gc*Chr_c+cvxInt_c) - Clr_c.T)
        else:
            fid_b = cvx.sum_squares(Gc*Chr_c - Clr_c)

        fidelity_list.append(fid_b)
    print '#fidelity_list', len(fidelity_list)
    cvxFidelityExp = sum(fidelity_list)

    ## Laplacian regularization
    cvxU = cvx.Constant(U)
    regLaplade_list = []
    vhrc = vhr*Nc
    for voxel in xrange(vhr):
        cvxLapaceReg = cvx.quad_form(cvxChr[voxel:vhrc:vhr], cvxU)
        regLaplade_list.append(cvxLapaceReg**2)
    cvxLaplaceRegExp = sum(regLaplade_list)
    
    
    ## 3D Tv-Norm Regularization
    cvxM = cvx.Constant(M)
    cvxC_byCoef = cvx.reshape(cvxChr, vhr, Nc)
    # (Nb,Nc)*(Nc,vhr) = (Nb, vhr).T = (vhr, Nb) 
    cvxYhr = cvx.reshape((cvxM*cvxC_byCoef.T).T, vhr*Nb, 1)
    print 'defining cvx3DTvNomExp', cvxYhr.size,Nx, Ny, Nz, Nb
    cvx3DTvNomExp = tvn.tv3d(cvxYhr, Nx, Ny, Nz, Nb)
    
    
    #Sparcity regularization
    cvxNorm1 = cvx.norm1(cvxChr)
    
    
    ## Mapl weight
    print 'animal', 'rata'
    beta = cvx.Parameter(value=parameters['beta'], name='beta', sign='positive')#3.197e-10
    print '   parameters[beta]', parameters['beta']
    ## Sparcity weight
    alpha = cvx.Parameter(value=parameters['alpha'], name='alpha', sign='positive')#4.865e-10
    print '   parameters[alpha]', parameters['alpha']
    ## Tv-norm weight
    gamma = cvx.Parameter(value=parameters['gamma'], name='gamma',sign='positive')
    print '   parameters[gamma]', parameters['gamma']
    ## Fidelity weight
    lamda = cvx.Parameter(value=parameters['lamda'], name='lamda',sign='positive')
    print '   parameters[lamda]', parameters['lamda']


    ### AS VARIABLES
    # beta = cvx.Variable(name='beta')
    # beta.value = 0.2
    ## Sparcity weight
    # alpha = cvx.Variable(name='alpha')
    # alpha.value = 4000
    ## Fidelity weight
    # gamma = cvx.Variable(name='gamma')
    # lamda = cvx.Variable(name='lamda')
    # lamda.value =0.5
    
    # Form objective.
    obj  = cvx.Minimize(lamda*cvxFidelityExp + beta*cvxLaplaceRegExp + alpha*cvxNorm1 + gamma*cvx3DTvNomExp)
    #obj = cvx.Minimize(lamda*cvxFidelityExp + beta*cvxLaplaceRegExp + alpha*cvxNorm1)
         
    # Constraints
    constraints = []
    #constraints = [lamda > 0 , alpha > 0, beta > 0]
    #constraints.append(cvxYhr >= 0)
    #Agregar q M*C es positivo o deberia

    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)


    return prob, cvxFidelityExp ,  cvxLaplaceRegExp , cvxNorm1, cvx3DTvNomExp

def find_closest_b(b, list_of_bs):
    dif = np.abs(b-list_of_bs[0])
    closest = list_of_bs[0]
    for elem in list_of_bs :
        if np.abs(b-elem) < dif :
            dif = np.abs(b-elem)
            closest = elem
    return closest


def define_problem_f2(i_lr, i_hr_shape, G, M, U, tau, gtab, scale, intercept=None):
    return None
    Nb, Nc = M.shape
    Nx, Ny, Nz, bval = i_hr_shape
    vlr = Nx * Ny * Nz / (scale ** 3)
    vlrb = vlr * bval
    vhr = Nx * Ny * Nz
    vhrb = vhr * bval

    ## Hr volumes
    Yhr = cvx.Variable(vhrb, 1, name='cvxYhr')
    # Yhr.value = np.ones((vhrb, 1))*i_lr.mean()

    ## MAPL params
    # cvxChr = cvx.Constant(C_hr.reshape(-1, order='F'))
    cvxChr = cvx.Variable(vhr * Nc, name='cvxChr')
    # M:(Nb,Nc)
    cvxMaplE = (M * cvx.reshape(cvxChr, vhr, Nc).T).T
    # Hr image in row by b-val
    YhrMapl = cvx.reshape(Yhr, vhr, bval)
    # Mapl dual expression
    cvxMaplDualExp = cvx.sum_squares(cvxMaplE - YhrMapl)

    ## Laplacian regularization
    cvxU = cvx.Constant(U)
    regLaplade_list = []
    vhrc = vhr * Nc
    for voxel in xrange(vhr):
        cvxLapaceReg = cvx.quad_form(cvxChr[voxel:vhrc:vhr], cvxU)
        regLaplade_list.append(cvxLapaceReg ** 2)
    cvxLaplaceRegExp = sum(regLaplade_list)

    ## LA FORMA MATRICIAL NO ME DEJA DICE QUE NO SE PUEDE MULTIPLICAR DOS MATRICES
    #  cvxChr:(vhrc, 1) U:(Nc,Nc)
    # cvxC_byCoef = cvx.reshape(cvxChr, vhr, Nc)
    # cvxCUC = cvx.diag(cvxC_byCoef*cvxU*cvxC_byCoef.T)
    # cvxLapaceRegExp = cvx.sum_squares(cvxCUC)


    ## Fidelity expression
    # cvxG = G
    fidelity_list = []
    lapace_list = []
    for i in xrange(Nb):
        b = gtab.bvals[i]
        b_offset_hr = i * vhr
        Yhr_b = Yhr[b_offset_hr:b_offset_hr + vhr]
        # Aprovecho para setearle un valor unicia
        b_close = b
        if b not in G.keys():
            b_close = find_closest_b(b , G.keys())
            print t4, "WARNING: bval=", b , ' not in G dict-of-matrix use bval=', b_close, 'instead'
            sys.stdout.flush()

        Gb = cvx.Constant(G[b_close])

        # Sometimes the data set has E(q) for a same q , we need just one (so we take the average of al samples)
        if i_lr[b].shape[1] > 1 :
            #print '> Este set de datos tiene mas de una vez el b=', b, 'repeticiones=', len([z for z in gtab.bvals if z == b ])
            Ylr_b = cvx.Constant(i_lr[b].mean(axis=1))
        else:
            Ylr_b = cvx.Constant(i_lr[b])

        if intercept is not None:
            cvxInt_b = cvx.Constant(intercept[b_close])
            # cvxInt_c:(vlr, 1)
            fid_b = cvx.sum_squares((Gb * Yhr_b + cvxInt_b) - Ylr_b)
        else:
            fid_b = cvx.sum_squares(Gb * Yhr_b - Ylr_b)
        fidelity_list.append(fid_b)
    cvxFidelityExp = sum(fidelity_list)

    ## 3D Tv-Norm Regularization
    cvx3DTvNomExp = tvn.tv3d(Yhr, Nx, Ny, Nz, Nb)

    # Sparcity regularization
    cvxNorm1 = cvx.norm1(cvxChr)

    ## Mapl weight
    beta = cvx.Parameter(value=1.452e-15, name='beta', sign='positive')  # 3.197e-10
    ## Sparcity weight
    alpha = cvx.Parameter(value=1.627e-15, name='alpha', sign='positive')  # 4.865e-10
    ## Fidelity weight
    lamda = cvx.Parameter(value=1., name='lamda', sign='positive')
    ## 3D-Tv weight
    gamma = cvx.Parameter(value=1.0e-15, name='gamma', sign='positive')
    ### AS VARIABLES
    # beta = cvx.Variable(name='beta')
    # beta.value = 0.2
    ## Sparcity weight
    # alpha = cvx.Variable(name='alpha')
    # alpha.value = 4000
    ## Fidelity weight
    # gamma = cvx.Variable(name='gamma')
    # lamda = cvx.Variable(name='lamda')
    # lamda.value =0.5

    # Form objective.
    # obj = cvx.Minimize(cvxFidelityExp + betha*cvxLapaceRegExp + alpha*cvx.norm(cvxChr) + gamma*cvx3DTvNomExp)
    #obj = cvx.Minimize(
    #    lamda * cvxFidelityExp + cvxMaplDualExp + beta * cvxLaplaceRegExp + alpha * cvxNorm1 + gamma * cvx3DTvNomExp)
    obj = cvx.Minimize(
        lamda * cvxFidelityExp + cvxMaplDualExp + beta * cvxLaplaceRegExp + alpha * cvxNorm1)

    # Constraints
    # constraints = [lamda > 0 , alpha > 0, beta > 0]
    constraints = [Yhr >= 0]
    # Agregar q M*C es positivo o deberia

    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)

    return prob, cvxFidelityExp, cvxLaplaceRegExp, cvxNorm1

import mymapl.minmapl as mapl
# In[6]:
def solveMin_fitCosnt(name_parameter, the_range, subject,i,j,k, loader_func, G, intercept=None, scale=2, verbose=False, prob=None):

    definition_fun = None
    if FORMULA == FORMULA_NO1 :
        # Get input for the subject to fit
        _, c_lr, gtab, i_hr, i_lr = samples.get_sample_of_mapl(subject,i,j,k, loader_func, bsize=BSIZE, scale=SCALE)
        c_lr = samples.split_by(c_lr)
        # Mapl params
        M, tau, mu, U = mapl.get_mapl_params2(gtab, radial_order=4)
        i_hr_fake = img_utils.downsampling2(i_lr, 0.5)
        print i_hr_fake.shape
        c_hr_initial = mapl.getC(i_hr_fake, gtab, radial_order=4)
        definition_fun = lambda toprint : define_problem_f1(
                                    c_lr,
                                    vhr,
                                    vlr,
                                    G,
                                    M, U,tau,
                                    gtab,
                                    scale,
                                    c_hr_initial=c_hr_initial,
                                    intercept=intercept,
                                    toprint=toprint)
    else:
        # Get input for the subject to fit
        i_hr, i_lr, gtab = samples.get_sample_of_dwi(subject, i,j,k,loader_func, bsize=BSIZE, scale=SCALE)
        #print t2, 'i_hr:', i_hr.shape, 'i_lr:', i_lr.shape
        i_lr = samples.split_by_bval(i_lr, gtab)
        # Mapl params
        M, tau, mu, U = mapl.get_mapl_params2(gtab, radial_order=4)

        definition_fun = lambda : define_problem_f2(i_lr, i_hr.shape, G, M, U, tau, gtab, scale, intercept=intercept)
    print 'i_hr', i_hr.shape, 'i_lr', i_lr.shape, 'bvals=', gtab.bvals.shape
    sys.stdout.flush()

    Nx, Ny, Nz, Nb = i_hr.shape
    Nb, Nc = M.shape
    nx, ny, nz = Nx / scale, Ny / scale, Nz / scale
    vhr, vlr = Nx * Ny * Nz, nx * ny * nz

    cvxFidelityExp, cvxLaplaceRegExp, cvxNorm1 = None, None, None

    b1000_index = indexs(gtab.bvals, 1000)
    b2000_index = indexs(gtab.bvals, 2000)
    b3000_index = indexs(gtab.bvals, 3000)

    measures = ['mse', 'mse1000', 'mse2000', 'mse3000']
    #info = dict((key, parray(base_folder + key + '_' + str(subject) + '.txt')) for key in measures)
    info = dict((key, []) for key in measures)
    
    info['cvxFidelityExp'] = []
    info['cvxLaplaceRegExp'] = []
    info['cvxNorm1'] = []
    info['cvx3DTvNomExp'] = []
    seg = 0

   
    """ Sequencial"""
    for i_val in xrange(len(the_range)) :
        val = the_range[i_val]
        _mse, _mse1000, _mse2000, _mse3000, seg, A, \
        cvxFidelityExp_val,  cvxLaplaceRegExp_val, \
        cvxNorm1_val, cvx3DTvNomExp_val = try_value(name_parameter, val, i_hr, M, Nx, Ny, Nz, Nb, Nc, b1000_index, b2000_index, b3000_index, definition_fun)
        info['mse'].append(_mse)
        info['mse1000'].append(_mse1000)
        info['mse2000'].append(_mse2000)
        info['mse3000'].append(_mse3000)

        info['cvxFidelityExp'].append(cvxFidelityExp_val)
        info['cvxLaplaceRegExp'].append(cvxLaplaceRegExp_val)
        info['cvxNorm1'].append(cvxNorm1_val)
        info['cvx3DTvNomExp'].append(cvx3DTvNomExp_val)


        seg += seg

        raise_warning_last_equal_result(info, the_range, name_parameter)

        if group_number_job == fit_index_job:
            print '$$ saving recontructed image of group', group_number_job, 'in', base_folder + 'A_g%d_val%d' % (group_number_job, i_val)
            np.save(rm.get_dir() + 'A_g%d_val%d' % (group_number_job, i_val), A)

    print 'info[mse]', info['mse']

    if group_number_job == fit_index_job:
        print '$$ saving original image of group', group_number_job, 'in', base_folder + 'i_hr_g%d' % (group_number_job)
        np.save(rm.get_dir() + 'i_hr_g%d' % (group_number_job), i_hr)

    print t3, 'fin fit al values for subject:', subject, 'segs:', seg,  datetime.datetime.now()
    # return A, C, seg, prob, cvxFidelityExp, cvxLaplaceRegExp , cvxNorm1, info
    return None, None, seg, None, None, None, None, info

def raise_warning_last_equal_result(info, the_range, name_parameter):
    last = len(info['mse']) - 1
    if  last > 1 :
        for sufijo in ['', '1000', '2000', '3000']:
            if info['mse' + sufijo][last] == info['mse'][last-1] :
                text = '***** info[mse] de param %s con valor %d y %d dan igual (%d)'%(name_parameter,the_range[last], the_range[last-1], info['mse'][last])
                warnings.warn(text)
                print text

def raise_warning_all_equal(res):
    for sufijo in ['', '1000', '2000', '3000']:
        current = res['mse' + sufijo]
        all_equal = True
        for i in xrange(len(current) - 1):
            if current[i] != current[i + 1]:
                all_equal = False
                break

        if all_equal:
            text = '***** mse' + sufijo + ' todos los valores iguales'
            warnings.warn(text)
            print text


def try_value(name_parameter, val, i_hr,M, Nx, Ny, Nz, Nb, Nc, b1000_index, b2000_index, b3000_index, definition_fun, max_iters=1000, verbose=False, res=None):
    print '****before define problem val=', val, '    ',  datetime.datetime.now()
    parameters[name_parameter] = val
    prob = None
    prob, cvxFidelityExp,  cvxLaplaceRegExp, cvxNorm1, cvx3DTvNomExp = definition_fun(name_parameter+'='+str(val))
    print 'id(prob)', id(prob)

    #parameters = dict( (v.name(), v) for v in prob.parameters())

    print t3, 'setting new ', name_parameter, '=',  parameters[name_parameter], '    ', datetime.datetime.now()
    sys.stdout.flush()

    max_its = MAXIT_BY_ROUND
    rounds = ROUNDS
    verbose = VERBOSE
    start_time = time.time()
    ant_status = prob.value
    for i in xrange(rounds):
        print 'it=', i, 'of', rounds,'max_iters=', max_its , datetime.datetime.now()
        prob.solve(solver='SCS', max_iters=max_its, eps=1.0e-05, verbose=verbose)  # Returns the optimal value.
        mejoro = 'mejoro' if prob.value < ant_status else 'no_mejoro'
        print t3, "--- status:", prob.status, "optimal value=", prob.value, mejoro,  'i_hr:', i_hr.shape, datetime.datetime.now()
        sys.stdout.flush()

        if cvxFidelityExp is not None:
            print t3, '>cvxFidelityExp', cvxFidelityExp.value

        if cvxLaplaceRegExp is not None:
            print t3, '>cvxLaplaceRegExp', cvxLaplaceRegExp.value

        if cvxNorm1 is not None:
            print t3, '>cvxNorm1', cvxNorm1.value, datetime.datetime.now()
	
	#time_normatv = time.time()
        #if cvx3DTvNomExp is not None:
        #    print t3, '>cvx3DTvNomExp', cvx3DTvNomExp.value, int((time.time() - time_normatv)/ 60), 'min',datetime.datetime.now()

        sys.stdout.flush()

        pval_ant = prob.value
    seg = time.time() - start_time
    minutes = int(seg / 60)
    print t3, "--- time of optimization : %d' %d'' (subject:%s, %s: %f) ---" % (minutes , seg%60, subject, name_parameter, val)
    print t3, "--- status:", prob.status, "optimal value", prob.value ,np.abs(pval_ant-prob.value),  datetime.datetime.now()
    sys.stdout.flush()

    # Get result
    variables = None
    variables = dict( (v.name(), v) for v in prob.variables())

    cvxChr = variables['cvxChr']
    C = None
    C = np.asarray(cvxChr.value, dtype='float32').reshape((Nx, Ny, Nz, Nc), order='F')
    print id(cvxChr),'@@@@@@@ C.mean=',  C.mean(),  datetime.datetime.now()

    A = None
    if 'cvxYhr' in variables :
        print 'Tomando cvxYhr',  datetime.datetime.now()
        cvxYhr = None
        cvxYhr = variables['cvxYhr']
        A = np.asarray(cvxYhr.value, dtype='float32').reshape((Nx, Ny, Nz, Nb), order='F')
    else:
        print 'tomando M*C',  datetime.datetime.now()
        A = M.dot(C.reshape((Nx*Ny*Nz, Nc), order='F').T).T
        A = A.reshape((Nx, Ny, Nz, Nb), order='F')

    cvxFidelityExp_val = -1
    if cvxFidelityExp is not None:
        cvxFidelityExp_val = cvxFidelityExp.value

    cvxLaplaceRegExp_val = -1
    if cvxLaplaceRegExp is not None:
        cvxLaplaceRegExp_val = cvxLaplaceRegExp.value

    cvxNorm1_val = -1
    if cvxNorm1 is not None:
        cvxNorm1_val = cvxNorm1.value

    print 'Antes de loguear normaTv',  datetime.datetime.now()
    sys.stdout.flush()

    cvx3DTvNomExp_val = -1
    if cvx3DTvNomExp is not None:
        cvx3DTvNomExp_val = cvx3DTvNomExp.value

    print 'Desp de loguear normaTv', datetime.datetime.now()
    sys.stdout.flush()

    _mse = ((A-i_hr)**2).mean()
    #info['mse'].append(mse)
    print t3, 'mse=', _mse, mse, datetime.datetime.now()

    _mse1000 = ((A[:, :, :, b1000_index]-i_hr[:, :, :, b1000_index])**2).mean()
    #info['mse1000'].append(mse1000)
    print t3, A[:, :, :, b1000_index].shape, i_hr[:, :, :, b1000_index].shape, 'mse1000=', _mse1000, datetime.datetime.now()
    #sys.stdout.flush()

    _mse2000 = ((A[:, :, :, b2000_index]-i_hr[:, :, :, b2000_index])**2).mean()
    #info['mse2000'].append(mse2000)

    _mse3000 = ((A[:, :, :, b3000_index]-i_hr[:, :, :, b3000_index])**2).mean()
    #info['mse3000'].append(mse3000)


    
    del (C, prob, cvxFidelityExp, cvxLaplaceRegExp, cvxNorm1, cvx3DTvNomExp)
    print t3, '.', datetime.datetime.now()
    #sys.stdout.flush()
    print t3, '.', datetime.datetime.now()
    print t3, '.'


    if res is not None:
        res[i] = (_mse, _mse1000, _mse2000, _mse3000, seg)

    return _mse, _mse1000, _mse2000, _mse3000, seg, A, cvxFidelityExp_val,  cvxLaplaceRegExp_val, cvxNorm1_val, cvx3DTvNomExp_val


def indexs(a, val):
    return [ i for i in xrange(a.size) if a[i] == val]



# In[8]:
def params_for(subjects, i, j, k, sample_maker, bvals_needed=None):
    ## The one that left out to validate

    if FORMULA == FORMULA_NO2:
        lr_samples, hr_samples = samples.buildT_grouping_by(subjects, i, j, k, sample_maker, use_bvals=True) #lr, hr
    else:
        lr_samples, hr_samples = samples.buildT_grouping_by(subjects, i, j, k, sample_maker)  # lr, hr

    # Build downsampling matrix
    print '= Training and fiting n_samples: %d ...' % len(subjects), datetime.datetime.now()
    regr, _ , _, intercept = e1f.train_grouping_by(hr_samples, lr_samples, intercept=INTERCEPT)

    G = dict((c,csr_matrix(regr[c].coef_)) for c in regr.keys())

    del(lr_samples)
    del(hr_samples)
    #gc.collect()

    return G, intercept
    
from conf_exp6 import *


if IS_NEF :
    RES_BASE_FOLDER = '/home/lgomez/workspace/iqt/results/'
else:
    RES_BASE_FOLDER = '/home/leexgo1987/Documentos/cs/inria/iqt/results'

# ## Solving the problem and cross-validation (leave one out)

if IS_NEF :
    formula_to_use = sys.argv[2]
else:
    formula_to_use = 'f1'

FORMULA = formulas[formula_to_use]

#bvals2000pos = [18, 27, 69, 75, 101, 107]

## Con imagenes pequenas multi-shel
SCALE=2
loader_func = hcp.load_subject_medium_noS0

if FORMULA == FORMULA_NO2:
    sample_maker = samples.get_sample_maker_of_dwi(loader_func, bsize=BSIZE)
else:
    sample_maker = samples.get_sample_maker_of_map(loader_func, bsize=BSIZE)

n_samples = len(subjects)

if IS_NEF :
    param_name = sys.argv[1]
else:
    param_name = 'lamda'

name_parameter = param_name
rango = params_range[param_name]

base_folder = RES_BASE_FOLDER + formula_to_use + '/' + param_name + '/'

# Metrics to save

mins_lamda   = []
times        = []

if IS_NEF :
    group_number_job = int(sys.argv[3])
else:
    group_number_job = 0

if IS_NEF :
    fit_index_job = int(sys.argv[4])%FITS
else:
    fit_index_job = 0

if IS_NEF :
    try:
        id_job = int(sys.argv[5])
    except IndexError:
        raise 'Falta parametro 5 (id_job:int)'
else:
    id_job = 1234

# Save the job descriptor
exp_name = 'exp6'
rm = ResultManager(RES_BASE_FOLDER  , 
                exp_name + '/' + formula_to_use + '/' + param_name,  
                id_job)
rm.add_data('params_range', dict((x[0], list(x[1])) for x in params_range.items()))
rm.add_data('name_parameter', name_parameter)
rm.add_data('formula', formula_to_use)
rm.add_data('scale', SCALE)
rm.add_data('intercept', INTERCEPT)
rm.add_data('max_its', MAXIT_BY_ROUND)
rm.add_data('rounds', ROUNDS)

# Optional
if IS_NEF :
    try:
        description = str(sys.argv[6])
        rm.add_data('description', description)        
    except IndexError:
        pass


print 'STARTING JOB', id_job,'FOR', param_name, 'USING FORMULA', FORMULA , ' GROUP-job:', group_number_job, 'FIT-index', fit_index_job,   datetime.datetime.now()
print 'WITH RANGE:', rango
print 'Intercept:', str(INTERCEPT)
sys.stdout.flush()



GROUPS = n_samples/GROUP_SIZE
RANGO= len(rango)

rm.add_data('n_samples', n_samples)
rm.save()

"""
mse = np.zeros((RANGO, FITS, GROUPS), dtype='float32')
mse1000 = np.zeros((RANGO, FITS, GROUPS), dtype='float32')
mse2000 = np.zeros((RANGO, FITS, GROUPS), dtype='float32')
mse3000 = np.zeros((RANGO, FITS, GROUPS), dtype='float32')
"""
mse = np.zeros((RANGO), dtype='float32')
mse1000 = np.zeros((RANGO), dtype='float32')
mse2000 = np.zeros((RANGO), dtype='float32')
mse3000 = np.zeros((RANGO), dtype='float32')

import copy

parameters = copy.copy(DEFAULT_PARAMETERS)


it = d.DmriPatchIterator(range(8, 12, 12), range(7, 14, 14), range(8, 12, 12))
for i, j, k in it:
    for group_num in [group_number_job]:
        subject_offset = GROUP_SIZE*group_number_job
        train_subjects = subjects[subject_offset:subject_offset+GROUP_SIZE]
        test_set = subjects[:subject_offset] + subjects[subject_offset+GROUP_SIZE:]
        test_set = test_set[:FITS]
        print 'len(test)', len(test_set), 'len(group)', len(train_subjects)

        # Linear regresion of this group
        print
        print datetime.datetime.now()
        train_time = time.time()
        G, intercept = params_for(train_subjects, i, j, k, sample_maker)
        train_time = time.time() - train_time
        print "== Training of Group:%d    (%d'%d'')"%(group_num, int(train_time/60), int(train_time%60)), datetime.datetime.now()

        sys.stdout.flush()

        for subject_index in [fit_index_job]:
            subject = test_set[subject_index]
            print '/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/'
            print t1, '== Group:%d of %d Fiting subject:%d(%d,%d,%d) of %d (%d)#' % (group_num, GROUPS, subject_index,i,j,k, FITS,  subject), datetime.datetime.now()
            print t1, '= Solving optimization problem (subject: %s, param: %s) === ' % (subject, param_name), datetime.datetime.now()
            sys.stdout.flush()

            A, C, seg, prob, cvxFidelityExp, cvxLaplaceRegExp, cvxNorm1, res =\
                solveMin_fitCosnt(name_parameter,
                                  rango,
                                  subject,
                                  i, j, k,
                                  loader_func,
                                  G,
                                  intercept=intercept,
                                  scale=SCALE,
                                  verbose=False)

            # Saving all results for analize latter
            mse[:] = res['mse']
            mse1000[:] = res['mse1000']
            mse2000[:] = res['mse2000']
            mse3000[:] = res['mse3000']


            raise_warning_all_equal(res)

            # Keeping the parameter value of each fitting that produce the min-mse (of all the val tested for the subjetc)
            index = np.argmin(np.array(res['mse']))
            print 'index del minimo:', index
            min_lamda = rango[index]
            mins_lamda.append(min_lamda)

            ## Para guardar info si se quiere
            #for key in res.keys():
            #    p = parray(base_folder + key + '_' + str(subject) + '.txt')
            #    p = p+res[key]

            times.append(seg)
            #optimal_vals.append(prob.value)

            if A is not None:
                del(prob, A, C, cvxFidelityExp, cvxLaplaceRegExp, cvxNorm1)

            del(seg)
            #print 'len(gc.garbage[:])', len(gc.garbage[:]),'number of unreachable objects found:', gc.collect()

        del(G, intercept)
        #gc.collect()


print 'Ultimos calculos', datetime.datetime.now()
sys.stdout.flush()

# Persist min vals
pmins_lamda   = parray(base_folder + 'mins_mses.txt', mins_lamda)
ptimes        = parray(base_folder +'times.txt',times)
#poptimal_vals = parray(base_folder +'optimal_vals.txt', optimal_vals)

# Log spended time
total_sec = np.array(times).sum()
print ' === TOTAL TIME :',  str(int(total_sec//60))+"'", str(int(total_sec%60))+ '"' , datetime.datetime.now()
sys.stdout.flush()

# Persist results
#if base_folder is not None: 
#    np.save(base_folder+ 'mins_alphas', mins_lamda)

## Saving mse's
base_name = rm.get_dir() + 'mse_g'+ str(group_number_job) +'_f'+str(fit_index_job)
np.save(base_name, mse)
print 'saved:', base_name
base_name = rm.get_dir() + 'mse%d_g'+ str(group_number_job) +'_f'+str(fit_index_job)
np.save(base_name%(1000), mse1000)
print 'saved:', base_name%(1000)
np.save(base_name%(2000), mse2000)
print 'saved:', base_name%(2000)
np.save(base_name%(3000), mse3000)
print 'saved:', base_name%(3000)

base_name = rm.get_dir() + '%s_g'+ str(group_number_job) +'_f'+str(fit_index_job)
terms = [
    'cvxFidelityExp',
    'cvxLaplaceRegExp',
    'cvxNorm1',
    'cvx3DTvNomExp'
]
for term in terms:
    np.save(base_name%(term), np.array(res[term]))
    print 'saved:', base_name%(term), np.array(res[term]).mean()



print 'Lito!', datetime.datetime.now()
sys.stdout.flush()
