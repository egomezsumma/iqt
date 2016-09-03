#!/user/lgomez/home/anaconda2/bin/python

import time
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import load.hcp_img_loader as hcp
from utils.persistance.persistence_array import parray
from scipy.sparse import csr_matrix
import experimento1_funciones as e1f
import load.samples as samples
import sys


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

# In[5]:
def define_problem_f1(c_lr, vhr, vlr, G, M, U, tau, gtab, scale, intercept=None):
    Nx, Ny, Nz = (12, 12, 12)#TODO: pasar
    Nb, Nc = M.shape
    
    ## LR volumes
    Clr = c_lr
    
    ## MAPL params
    #cvxChr = cvx.Constant(C_hr.reshape(-1, order='F'))
    
    cvxChr = cvx.Variable(vhr*Nc, name='cvxChr')
    ChrValueInitial = np.ones((vhr*Nc, 1), dtype='float32')
    for c in xrange(Nc): 
        c_offset_hr = c*vhr
        ChrValueInitial[c_offset_hr:c_offset_hr+vhr] = ChrValueInitial[c_offset_hr:c_offset_hr+vhr]*Clr[c].mean()
    cvxChr.value = ChrValueInitial
    
    ## Fidelity expression
    cvxG = G
    fidelity_list = []
    lapace_list = []
    for c in xrange(Nc):
        c_offset_hr = c*vhr
        Chr_c = cvxChr[c_offset_hr:c_offset_hr+vhr]
        # Aprovecho para setearle un valor unicia
        Gc = cvx.Constant(G[c])
        Clr_c = cvx.Constant(Clr[c])
        #Clr_c = cvx.Variable(Clr[c].shape[0], Clr[c].shape[1], name='Clr_'+str(c))

        #       Gc:(216:vlr, 1728:vhr) Chr_c:(1728:vhr, 1) Clr_c:(1, 216:vlr)
        if intercept is not None:
            cvxInt_c = cvx.Constant(intercept[c])
            #cvxInt_c:(vlr, 1)
            fid_b = cvx.sum_squares((Gc*Chr_c+cvxInt_c) - Clr_c.T)
        else:
            fid_b = cvx.sum_squares(Gc*Chr_c - Clr_c.T)
        
        fidelity_list.append(fid_b)    
    #cvxNc = cvx.Constant(Nc)
    #cvxFidelityExp = cvx.inv_pos(cvxNc)*sum(fidelity_list)
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
    cvxC_byCoef = cvx.reshape(cvxChr, vhr, Nc)
    # (Nb,Nc)*(Nc,vhr) = (Nb, vhr).T = (vhr, Nb) 
    cvxYhr = cvx.reshape((M*cvxC_byCoef.T).T, vhr*Nb, 1)
    #vx3DTvNomExp = tvn.tv3d(cvxYhr, Nx, Ny, Nz, Nb)
    
    
    #Sparcity regularization
    cvxNorm1 = cvx.norm1(cvxChr)
    
    
    ## Mapl weight
    beta = cvx.Parameter(value=3*1.452e-15, name='beta', sign='positive')#3.197e-10
    ## Sparcity weight
    alpha = cvx.Parameter(value=1.627e-15, name='alpha', sign='positive')#4.865e-10
    ## Tv-norm weight
    gamma = cvx.Parameter(value=0.05, name='gamma',sign='positive')
    ## Fidelity weight
    lamda = cvx.Parameter(value=1., name='lamda',sign='positive')

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
    #obj = cvx.Minimize(cvxFidelityExp + betha*cvxLapaceRegExp + alpha*cvx.norm(cvxChr) + gamma*cvx3DTvNomExp)
    obj = cvx.Minimize(lamda*cvxFidelityExp + beta*cvxLaplaceRegExp + alpha*cvxNorm1)
         
    # Constraints
    #constraints = [lamda > 0 , alpha > 0, beta > 0]
    constraints = [cvxYhr >= 0]
    #Agregar q M*C es positivo o deberia

    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    
    return prob, cvxFidelityExp ,  cvxLaplaceRegExp , cvxNorm1


# In[6]:
def solveMin_fitCosnt(name_parameter, the_range, c_lr, i_hr, G, M, U, tau, gtab, intercept=None, scale=2, max_iters=1500, verbose=False, prob=None):
    Nx, Ny, Nz, Nb = i_hr.shape
    Nb, Nc = M.shape
    nx, ny, nz = Nx/scale, Ny/scale, Nz/scale 
    vhr, vlr = Nx*Ny*Nz, nx*ny*nz 
    
    cvxFidelityExp,  cvxLaplaceRegExp, cvxNorm1 = None, None, None
        
    b1000_index = indexs(gtab.bvals, 1000)
    b2000_index = indexs(gtab.bvals, 2000)
    b3000_index = indexs(gtab.bvals, 3000)
    
    base_folder = RES_BASE_FOLDER + name_parameter + '/'
    measures = ['mse', 'mse1000', 'mse2000', 'mse3000']
    info = dict((key, parray( base_folder + key + '_'+subject+'.txt')) for key in measures)
    
    #info = {'mse':parray(RES_BASE_FOLDER+'mse_'+the), 'mse1000':[],'mse2000':[],'mse3000':[]}
    for val in the_range :
        prob, cvxFidelityExp ,  cvxLaplaceRegExp , cvxNorm1 = define_problem_f1(
                                    c_lr, 
                                    vhr, 
                                    vlr,
                                    G, 
                                    M, U,tau,
                                    gtab,
                                    scale,
                                    intercept=intercept)

        parameters = dict( (v.name(), v) for v in prob.parameters())
        parameters[name_parameter].value = val
    
        start_time = time.time()
        res = prob.solve(solver='SCS', max_iters=max_iters, eps=1.0e-05, verbose=verbose )  # Returns the optimal value.
        #res = prob.solve(solver='ECOS')  # Returns the optimal value.

        seg = time.time() - start_time

        minutes = int(seg / 60)
        print
        print
        print("--- time of optimization : %d' %d'' (subject:%s, %s: %f) ---" % (minutes , seg%60, subject, name_parameter, val))
        print "--- status:", prob.status, "optimal value", prob.value
        print 

        variables = dict( (v.name(), v) for v in prob.variables())
        #parameters = dict( (v.name(), v) for v in prob.parameters())
        #print variables

        cvxChr = variables['cvxChr']

        C = np.asarray(cvxChr.value, dtype='float32').reshape((Nx, Ny, Nz, Nc), order='F')

        A = M.dot(C.reshape((Nx*Ny*Nz, Nc), order='F').T).T
        A = A.reshape((Nx, Ny, Nz, Nb), order='F')

        mse = ((A-i_hr)**2).mean()
        info['mse'].append(mse)
        
        mse = ((A[:, :, :, b1000_index]-i_hr[:, :, :, b1000_index])**2).mean()
        info['mse1000'].append(mse)
        
        mse = ((A[:, :, :, b2000_index]-i_hr[:, :, :, b2000_index])**2).mean()
        info['mse2000'].append(mse)
        
        mse = ((A[:, :, :, b3000_index]-i_hr[:, :, :, b3000_index])**2).mean()
        info['mse3000'].append(mse)
        
        if cvxFidelityExp is not None:
            print 'cvxFidelityExp', cvxFidelityExp.value
        if cvxLaplaceRegExp is not None:
            print 'cvxLaplaceRegExp', cvxLaplaceRegExp.value
        if cvxNorm1 is not None:
            print 'cvxNorm1', cvxNorm1.value 

    return A, C, seg, prob, cvxFidelityExp, cvxLaplaceRegExp , cvxNorm1, info

def indexs(a, val):
    return [ i for i in xrange(a.size) if a[i] == val]


# In[8]:
def params_for(the_one_out, sample_maker, n_samples, loader_func, scale=2):
    ## The one that left out to validate
    
    i_hr, i_lr, gtab = samples.get_sample_of_dwi(the_one_out, subjects, loader_func, scale=scale)
    #C_hr, _, _ = samples.get_sample_of_mapl(the_one_out, subjects, loader_func, scale=scale)
    
    ### Aca shiftear el arreglo de sujetos (train deja el ultimo afuera del entrenamiento)
    lr_samples, hr_samples = samples.buildT_grouping_by(sample_maker, n_samples) #lr, hr

    # Build downsampling matrix
    print '= Training and fiting n_samples: %d ...' % (n_samples)
    regr, chr_train , clr_train, chr_test, clr_test, intercept =                 e1f.train_grouping_by(hr_samples, lr_samples, intercept=True)
    del(chr_train)
    del(clr_train)
    del(chr_test)

    G = dict((c,csr_matrix(regr[c].coef_)) for c in regr.keys())
    
    # Mapl params
    M, tau, mu, U = mapl.get_mapl_params2(gtab, radial_order = 4)

    return i_hr, i_lr, gtab, G,intercept, M, tau, mu, U , clr_test
    


# ## Solving the problem and cross-validation (leave one out)
RES_BASE_FOLDER = './resultados/exp6/'
VMIN, VMAX=0, 1

voi_hr_shape = (12, 12, 12, 6)
voi_lr_shape = (6, 6, 6, 6)

subjects = [100307, 100408, 180129, 180432, 180836, 180937]
#subjects = [100307, 100408, 180129, 180432]

bvals2000pos = [18, 27, 69, 75, 101, 107]

## Con imagenes pequenas multi-shel
loader_func = hcp.load_subject_medium_noS0
sample_maker = samples.get_sample_maker_of_map(subjects, loader_func, scale=2)

n_samples = 3
iterations = 3


param_name = sys.argv[1]
params_range = {
    'lamda': np.arange(0.001, 0.4, 0.1),
    'alpha': np.arange(0.001, 0.4, 0.1),
    'beta': np.arange(0.001, 0.4, 0.1),
    'gamma': np.arange(0.001, 0.4, 0.1)
}

name_parameter = param_name
rango = params_range[param_name]
print 'STARTING JOB FOR', param_name, 'WITH RANGE:', rango


base_folder = RES_BASE_FOLDER + '/' + param_name

# Metrics to save
mins_lamda   = parray(base_folder + '/mins_mses.txt')
times        = parray(base_folder +'/times.txt')
optimal_vals = parray(base_folder +'/optimal_vals.txt')



for i in range(0, iterations):
    subjects.append(subjects.pop(0))
    subject = str(subjects[len(subjects)-1])
    print '== Leaving out: #', subject
    the_one_out = len(subjects)-1
    
    i_hr, i_lr, gtab, G, intercept, M, tau, mu, U , clr_test =\
        params_for(the_one_out, sample_maker, n_samples, loader_func, scale=2)

    print
    print
    print
    print '= Solving optimization problem (subject:%s, param:%s) === ' % (subject, param_name)

    A, C, seg, prob, cvxFidelityExp, cvxLaplaceRegExp, cvxNorm1, res =\
        solveMin_fitCosnt(name_parameter,
                          rango,  
                          clr_test, 
                          i_hr, 
                          G, 
                          M, U, tau, 
                          gtab, 
                          intercept=intercept, 
                          scale=2, 
                          max_iters=10, 
                          verbose=False)

    index = np.argmin(np.array(res['mse']))
    min_lamda = rango[index]
    mins_lamda.append(min_lamda)

    times.append(seg)
    optimal_vals.append(prob.value)

mins_lamda = mins_lamda.asnumpy()

# Log spended
total_sec = np.array(times).sum()
print ' === TOTAL TIME :',  str(int(total_sec//60))+"'", str(int(total_sec%60))+ '"'

# Persist results
#if base_folder is not None: 
#    np.save(base_folder+ 'mins_alphas', mins_lamda)

print 'mean=', mins_lamda.mean(), mins_lamda
plt.bar(xrange(mins_lamda.size), mins_lamda)
plt.savefig(base_folder + '/mins_' + param_name + '.pdf')

# In[11]:
print 'rangos:', rango
print 'mins_%s:' % (param_name) , mins_lamda

#dict((v.name(), v.value) for v in prob.variables())
print 'Lito!'
