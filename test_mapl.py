from mapmri import mapmri
from dipy.viz import fvtk
from dipy.data import fetch_cenir_multib, read_cenir_multib, get_sphere
from dipy.core.gradients import gradient_table
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import time
import numpy as np
import matplotlib.pyplot as plt
from utils import img_utils
import seaborn as sns
import utils.math_utils as mu
import nibabel as nib


def load_subject(index, numbers, bval=None, bvalpos=None):
    subject = str(numbers[index])
    folder = './HCP/' + subject + '/'
    bvals = np.loadtxt(folder + 'bvals_' + subject)
    bvecs = np.loadtxt(folder + 'bvecs_' + subject)

    if bvalpos is not None:
        img = nib.load(folder + 'data_small_12x12x12x6_' + subject + '_b' + str(bval) + '.nii.gz')
        gtab = gradient_table(bvals=bvals[bvalpos], bvecs=bvecs[:, bvalpos])
    else:
        img = nib.load(folder + 'data_small_12x12x12x6_' + subject + '.nii.gz')
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    return img, gtab


img, gtab = load_subject(0, [100307])
data = img.get_data()
data_small = data

gtab = gradient_table(bvals=gtab.bvals[0:6], bvecs=gtab.bvecs[0:6])

print('data.shape (%d, %d, %d, %d)' % data.shape)





radial_order = 6
reload(mapmri)
map_model_positivity_aniso = mapmri.MapmriModel(gtab,
                                                radial_order=radial_order,
                                                laplacian_regularization=False,
                                                positivity_constraint=True)

#mapfit_laplacian_aniso, Q, p, G, h, A, b
res = map_model_positivity_aniso.fit(data_small)
res.predict(gtab)
print res





