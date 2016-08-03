from dipy.sims import phantom


class DmriPatchRef(object):
    def __init__(self, img, indexs):
        self.img = img;
        self.indexs = indexs

    def get_volume(self):
        data=self.img;
        (x0, xf, y0, yf, z0, zf, b0, bf) = self.indexs
        return data[x0:xf, y0:yf, z0:zf, b0:bf]

    def get_indexs(self):
        return self.indexs



class NoisedDmriPatchRef(object):
    """
        Cada vez que se pide el volumne representado
        devuleve uno con diferntente ruido

        @see dipy.sims.phantom.add_noise
    """
    def __init__(self, img, indexs, snr=20, noise_type='rician'):
        self.img = img;
        self.indexs = indexs
        self._snr = snr
        self._noise_type = noise_type
        print "asas",self._snr, self._noise_type

    def get_volume(self):
        return self._get_volume_noised()

    def get_indexs(self):
        return self.indexs

    def _get_volume_noised(self):
        data = self.img;
        (x0, xf, y0, yf, z0, zf, b0, bf) = self.indexs
        img = data[x0:xf, y0:yf, z0:zf, b0:bf]
        print self._snr, self._noise_type
        noised_img = phantom.add_noise(img, snr=self._snr, noise_type=self._noise_type)
        return noised_img


class DmriPatch(object):
    def __init__(self, volume, indexs=None):
        self.volume = volume
        self.indexs = indexs

    def get_volume(self):
        return self.volume

    def get_indexs(self):
        return self.indexs