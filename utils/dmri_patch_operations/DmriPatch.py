
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


class DmriPatch(object):
    def __init__(self, volume, indexs=None):
        self.volume = volume;
        self.indexs = indexs

    def get_volume(self):
        return self.volume

    def get_indexs(self):
        return self.indexs