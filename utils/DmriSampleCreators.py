from DmriVolumesRandomIndexers import DmriLrHrCubicPatchVolumeRandomIndexer
from utils.dmri_patch_operations.DmriPatch import DmriPatchRef
from utils.img_utils import downsampling
from dipy.sims import phantom


#
#  if scale=2 --> la mitad
#  if scale=0.5 --> el doble
class LrHrDmriRandomSampleCreator(object):
    """
        Dada una DownsampledImage devuelve objetos DmriPatch
        segun la estrategia pasada como parametros
    """
    def __init__(self, lr_hr_img, n, m, lr_hr_vol_it=None):
        ''' > lr_hr_img: Objeto con imagenes en LR y HR
            > n, m: parametros para el tamano del patch '''
        self._name = lr_hr_img.name
        self.img_hr_data = lr_hr_img.get_hr_img()
        self.gtab = lr_hr_img.get_gtab()
        self.img_lr = lr_hr_img.get_lr_img()
        self.lr_hr_vol_it = lr_hr_vol_it
        if self.lr_hr_vol_it is None:
            self.lr_hr_vol_it = DmriLrHrCubicPatchVolumeRandomIndexer(self.img_lr.shape, n, m)

    @property
    def name(self):
        return self._name

    def get_gtab(self):
        return self.gtab;

    def get_lr_img(self):
        return self.img_lr

    def get_hr_img(self):
        return self.img_hr_data

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        indices = self.lr_hr_vol_it.next()
        lr_patch = DmriPatchRef(self.get_lr_img(), indices['lr'])
        hr_patch = DmriPatchRef(self.get_hr_img(), indices['hr'])
        return lr_patch, hr_patch

    def size(self):
        return self.lr_hr_vol_it.size()



class NoisedLrHrDmriSampleCreator(LrHrDmriRandomSampleCreator):
    """
        Dada una DownsampledImage devuelve objetos DmriPatch
        segun la estrategia pasada como parametros pero con un cierto
        ruido que puede ser 'rician' (default) o 'gausean'

        @see dipy.sims.phantom.add_noise
    """

    def __init__(self, lr_hr_img, n, m, lr_hr_vol_it=None, snr=20, noise_type='rician'):
        super(NoisedLrHrDmriSampleCreator, self).__init__(lr_hr_img, n, m, lr_hr_vol_it)
        self._snr = snr
        self._noise_type = noise_type

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        indices = self.lr_hr_vol_it.next()
        print indices
        lr_noised = self._get_volume_noised(self.get_lr_img(), indices['lr'])
        hr_noised = self._get_volume_noised(self.get_lr_img(), indices['hr'])
        lr_patch = DmriPatchRef(lr_noised, indices['lr'])
        hr_patch = DmriPatchRef(hr_noised, indices['hr'])
        return lr_patch, hr_patch

    def _get_volume_noised(self, data, indexs):
        (x0, xf, y0, yf, z0, zf, b0, bf) = indexs
        img = data[x0:xf, y0:yf, z0:zf, b0:bf]
        noised_img = phantom.add_noise(img, snr=self.self._snr,noise_type=self._noise_type)
        return noised_img

    def size(self):
        return self.lr_hr_vol_it.size()



