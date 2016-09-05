import numpy as np


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
    'lamda': np.arange(0.2, 2.0, 2),#9
    'alpha': np.arange(1.627e-15, 2.0, 0.2),#10
    'beta': np.arange(1.452e-15, 1.452e-14, 1.452e-15),#10
    'gamma': np.arange(0.05, 0.9, 0.09) #10
}

voi_hr_shape = (12, 12, 12, 6)
voi_lr_shape = (6, 6, 6, 6)

FITS =1#11
GROUP_SIZE=3#5

