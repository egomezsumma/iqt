from DmriVolumesRandomIndexers import DmriLrHrCubicPatchVolumeRandomIndexer
from utils.img_utils import downsampling


#
#  if scale=2 --> la mitad
#  if scale=0.5 --> el doble
class LrHrDmriRandomSampleCreator(object):

    def __init__(self, img_hr, n, m):
        self.img_hr = img_hr
        self.scale = m
        self.img_lr, self.lr_affine = downsampling(img_hr, m)
        self.lr_hr_vol_it = DmriLrHrCubicPatchVolumeRandomIndexer(self.img_lr.shape, n, m)

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        indexs  = self.lr_hr_vol_it.next()
        lr_patch = DmriPatch(self.img_lr, indexs['lr'])
        hr_patch = DmriPatch(self.img_hr, indexs['hr'])
        return lr_patch, hr_patch


class DmriPatch(object):
    def __init__(self, img, indexs):
        self.img = img;
        self.indexs = indexs

    def get_volume(self):
        data=self.img.get_data()
        (x0, xf, y0, yf, z0, zf, bi) = self.indexs
        return data[x0:xf, y0:yf, z0:zf, bi]
