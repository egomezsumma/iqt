import numpy as np
import sys


RES_BASE_FOLDER = '/home/lgomez/workspace/iqt/results/exp6/'
VMIN, VMAX=0, 1
BSIZE=55

FORMULA_NO1 = 'Formula1'
FORMULA_NO2 = 'Formula2'

formulas = {
    'f1': FORMULA_NO1,
    'f2': FORMULA_NO2
}

params_range = {
    'lamda': np.arange(1.0, 20.0, 2.0),#10
    #'lamda': np.arange(0.2, 2.0, 0.2)
    'alpha': np.arange(1.627e-15, 0.5, 0.05),#10
    'beta': np.arange(1.452e-15, 20, 2),#10
    #'beta': np.arange(1.452e-15, 1.452e-14, 1.452e-15),#10
    'gamma': np.arange(0.05, 0.9, 0.09)#10
}


voi_hr_shape = (12, 12, 12, 6)
voi_lr_shape = (6, 6, 6, 6)


IS_NEF = '/home/lgomez/' in sys.prefix


if IS_NEF :
    subjects = list(np.loadtxt('/home/lgomez/demo/50sujetos.txt', dtype='int'))
else:
    subjects = [100307, 100408, 180129, 180432, 180836, 180937]
    #subjects = [100307, 100408, 180129, 180432]

MAX_ITERS=1000

FITS =1#11
GROUP_SIZE=5

INTERCEPT=False


