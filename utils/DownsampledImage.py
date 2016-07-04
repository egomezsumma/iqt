
from utils.img_utils import downsampling

# Simplemente toma una imagen la downsamplea segun parametro 'scale'
# y se guarda una referencia a (gtab-original, data-original, data-downsampleada)
#  if scale=2 --> la mitad
#  if scale=0.5 --> el doble
class DownsampledImage(object):

    def __init__(self, name, img_hr_data, gtab,  scale):
        self._name = name;
        self.img_hr_data = img_hr_data
        self.gtab = gtab;
        self.scale = scale;
        self.img_lr, self.lr_affine = downsampling(img_hr_data, scale)

    @property
    def name(self):
        return self._name

    def get_gtab(self):
        return self.gtab;

    def get_lr_img(self):
        return self.img_lr

    def get_hr_img(self):
        return self.img_hr_data.get_data()