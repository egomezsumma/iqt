#import numpy as np;
from utils.dmri_patch_operations.DtiPatch import DtiPatch
from utils.dmri_patch_operations.DmriPatch import DmriPatch
import dipy.reconst.dti as dti


# Es de una imagen en particular
class DtiModel(object):

    def __init__(self, gtab):
        self.gtab=gtab
        self._tenmodel = None #lazy inicialization

    def _fit_model(self, volume):

        # img contains a nibabel Nifti1Image object (with the data) and gtab contains a GradientTable object (information about the gradients e.g. b-values and b-vectors).

        data = volume

        #from dipy.segment.mask import median_otsu
        #maskdata, mask = median_otsu(data, 3, 1, True,
        #                             vol_idx=range(10, 50), dilate=2)

        if self._tenmodel is None:
            self._tenmodel = dti.TensorModel(self.gtab)
        #          import dipy.denoise.noise_estimate as ne
        #          sigma = ne.estimate_sigma(data)
        #          dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)

        tenfit = self._tenmodel.fit(data, mask=data[..., 0] > 200)

        return tenfit.lower_triangular()

    def _predict_model(self, dti_patch):
        dti_lo_tri_vol = dti_patch.get_volume()
        dti_params_vol = dti.eig_from_lo_tri(dti_lo_tri_vol)
        dmri_data_vol = self._tenmodel.predict(dti_params_vol)
        return dmri_data_vol


    # f: Dmri -> Dti
    # debe ser un volumne que pertenesca a la gtab pasada en el init
    # retur: DtiPatch
    def get_dti_params(self, dmri_patch):
        return DtiPatch(self._fit_model(dmri_patch.get_volume()), dmri_patch.get_indexs())

    # g: Dti -> Dmri
    # return: DmriPatch (con los indices de la imagen original)
    def get_signal(self, dti_patch):
        dmri_data_vol = self._predict_model(dti_patch)
        return DmriPatch(dmri_data_vol, dti_patch.get_indexs())

