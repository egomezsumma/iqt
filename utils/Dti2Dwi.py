

import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
from utils.DataGetter import DataGetter


class Dti2Dwi(object):



    """
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
    """
    def __init__(self, dti_volume):
        self._lt_indices = np.array([[0, 1, 3],
                                [1, 2, 4],
                                [3, 4, 5]])

        size = dti_volume[0][0][0].size
        if size == 6 :
            self._dti = self.from_lower_triangular(dti_volume)
        elif size == 9:
            self._dti = dti_volume

    def from_lower_triangular(self, D):
        """ Returns a tensor given the six unique tensor elements
        Given the six unique tensor elments (in the order: Dxx, Dxy, Dyy, Dxz, Dyz,
        Dzz) returns a 3 by 3 tensor. All elements after the sixth are ignored.
        Parameters
        -----------
        D : array_like, (..., >6)
            Unique elements of the tensors
        Returns
        --------
        tensor : ndarray (..., 3, 3)
            3 by 3 tensors
        """
        return D[..., self._lt_indices]


    def predict(self, gtab, S0=1, step=None):
        dti = self._dti
        sx, sy, sz = dti.shape[0:3]

        bvals = gtab.bvals
        bvecs = gtab.bvecs
        bvals_size = bvals.size

        B = self._calculateB(bvals, bvecs)
        dwi = np.zeros((sx,sy,sz, bvals_size), dtype='float')
        for i in range(0, sx):
            for j in range(0, sy):
                for k in range(0, sz):
                    dti_ijk = dti[i][j][k]
                    #D = dti_ijk.quadratic_form
                    D = dti_ijk

                    BD = np.dot(B, D)
                    #dwi[i][j][k] = np.zeros(bvals_size, dtype=float)
                    for z in range(0, bvals_size):
                        dwi[i][j][k][z] = np.exp((-1)*np.dot(BD[z], B[z])) * S0[i][j][k]
        return dwi


    def _calculateB(self, bvals, bvecs):
        size, _ = bvecs.shape
        res = np.zeros((size, 3), dtype='float')
        for i in range(0, size):
            res[i] = bvals[i] * bvecs[i]
        return res;


    def save_nifti(name, data, affine):
        nifti1img = nib.Nifti1Image(data, affine)
        nib.save(nifti1img, name + '.nii.gz')
