from utils.dmri_patch_operations.DmriPatch import DmriPatchRef, NoisedDmriPatchRef


#
#  if scale=2 --> la mitad
#  if scale=0.5 --> el doble
class LrHrDmriRandomSampleCreator(object):
    """
        Dada una DownsampledImage devuelve objetos DmriPatch
        segun la estrategia pasada como parametros

        =[COMPATIVILIDAD PARA ATRAS]:==========================================
        Equivalente llamarlo
            antes :
                LrHrDmriRandomSampleCreator(lr_hr_img, n, m)
            ahora
                lr_hr_vol_it = DmriLrHrCubicPatchVolumeRandomIndexer(self.img_lr.shape, n, m)
                LrHrDmriRandomSampleCreator(lr_hr_img, lr_hr_vol_it)
        ========================================================================
    """
    def __init__(self, lr_hr_img, lr_hr_vol_it):
        self._name = lr_hr_img.name
        self.img_hr_data = lr_hr_img.get_hr_img()
        self.gtab = lr_hr_img.get_gtab()
        self.img_lr = lr_hr_img.get_lr_img()
        self.lr_hr_vol_it = lr_hr_vol_it

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
        Idem 'LrHrDmriRandomSampleCreator' pero devuelve patch
        del tipo 'NoisedDmriPatchRef' con un cierto
        ruido que puede ser 'rician' (default) o 'gausean'

        @see dipy.sims.phantom.add_noise
        @see utils.dmri_patch_operations.DmriPatch.NoisedDmriPatchRef
    """

    def __init__(self, lr_hr_img, lr_hr_vol_it, snr=20, noise_type='rician'):
        super(NoisedLrHrDmriSampleCreator, self).__init__(lr_hr_img, lr_hr_vol_it)
        self._snr = snr
        self._noise_type = noise_type

    def next(self):
        indices = self.lr_hr_vol_it.next()
        print 'asasasasasasasasasadadad'
        lr_patch = NoisedDmriPatchRef(self.get_lr_img(), indices['lr'])
        hr_patch = NoisedDmriPatchRef(self.get_hr_img(), indices['hr'])
        return lr_patch, hr_patch


