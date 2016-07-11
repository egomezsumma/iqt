import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
from utils.DataGetter import DataGetter

from utils.Dti2Dwi import Dti2Dwi


def calculateB(bvals, bvecs):
    size, _ = bvecs.shape
    res = np.zeros((size, 3), dtype='float')
    for i in range(0, size):
        res[i] = bvals[i]*bvecs[i]
    return res;


def save_nifti(name, data, affine):
    nifti1img = nib.Nifti1Image(data, affine)
    nib.save(nifti1img, name+ '.nii.gz')


d = DataGetter()
datas_names = [
        DataGetter.STANDFORD_HARDI_DATA,
    ]
datas = d.get_data(datas_names)

img = datas[DataGetter.STANDFORD_HARDI_DATA]['img']
gtab = datas[DataGetter.STANDFORD_HARDI_DATA]['gtab']

dwi_orig = img.get_data()
tenmodel = dti.TensorModel(gtab)
dti = tenmodel.fit(dwi_orig, mask=dwi_orig[..., 0] > 200)

######

dti2dwi=Dti2Dwi(dti.lower_triangular())
S0 = dwi_orig[:,:,:,0]
dwi3 = dti2dwi.predict(gtab,S0)
save_nifti('prueba_reconstruccion3', dwi3, img.affine)

#########

sx, sy, sz, bvals_size = img.shape

bvals = gtab.bvals
bvecs = gtab.bvecs

B = calculateB(bvals, bvecs)
dwi = np.zeros((sx,sy,sz, bvals_size), dtype='float')
dwi2 = np.zeros((sx,sy,sz, bvals_size), dtype='float')
for i in range(0, sx):
    for j in range(0, sy):
        for k in range(0, sz):
            dti_ijk = dti[i][j][k]
            D = dti_ijk.quadratic_form

            #if np.any(D):
            #    print 'hola'

            BD = np.dot(B, D)

            #dwi[i][j][k] = np.zeros(bvals_size, dtype=float)
            for z in range(0, bvals_size):
                dwi[i][j][k][z] = np.exp((-1)*np.dot(BD[z], B[z]))
                dwi2[i][j][k][z] = dwi[i][j][k][z] * dwi_orig[i][j][k][0]

                """
                if np.linalg.norm(dwi2[i][j][k][z] - dwi_orig[i][j][k][z])  > 0:
                    print 'aca'

                if np.any(dwi[i][j][k]):
                    print 'aca'

                if np.any(dwi_orig[i][j][k]):
                    print 'aca'
                """

save_nifti('prueba_reconstruccion', dwi, img.affine)

save_nifti('prueba_reconstruccion2', dwi2, img.affine)

save_nifti('prueba_reconstruccion_orig', dwi_orig, img.affine)

for i in range(0, sx):
    for j in range(0, sy):
        for k in range(0, sz):
            if np.any(dwi[i][j][k]) or np.any(dwi_orig[i][j][k]) :
                dif = np.abs(dwi[i][j][k]-dwi_orig[i][j][k])