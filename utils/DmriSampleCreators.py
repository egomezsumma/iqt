from DmriVolumesRandomIndexers import DmriLrHrCubicPatchVolumeRandomIndexer
from utils.dmri_patch_operations.DmriPatch import DmriPatchRef
from utils.img_utils import downsampling


#
#  if scale=2 --> la mitad
#  if scale=0.5 --> el doble
class LrHrDmriRandomSampleCreator(object):

    def __init__(self,lr_hr_img,  n, m):
        self._name = lr_hr_img.name;
        self.img_hr_data = lr_hr_img.get_hr_img();
        self.gtab = lr_hr_img.get_gtab();
        self.img_lr = lr_hr_img.get_lr_img()
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
        indexs  = self.lr_hr_vol_it.next()
        lr_patch = DmriPatchRef(self.get_lr_img(), indexs['lr'])
        hr_patch = DmriPatchRef(self.get_hr_img(), indexs['hr'])
        return lr_patch, hr_patch

    def size(self):
        return self.lr_hr_vol_it.size()

