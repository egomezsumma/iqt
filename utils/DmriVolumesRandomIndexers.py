from utils.RandomRanges import Fixed1DRange, Random3DRange, All3DRangePosibleNotOverlapping

class DmriVolumeRandomIndexer(object):

    def __init__(self, img_shape, volume_strategy=None, bval_strategy=None):
        self.img_shape = img_shape
        self.volume_strategy = volume_strategy if volume_strategy is not None else self._get_default_volume_strategy(img_shape);
        self.bval_strategy = bval_strategy if bval_strategy is not None else Fixed1DRange(0,img_shape[3]);


    def _get_default_volume_strategy(self, img_shape):
        shape = img_shape
        patches = 2
        return Random3DRange(shape[0:3],shape[0]//patches,shape[1]//patches, shape[2]//patches)

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        (x0, xf, y0, yf, z0, zf) = self.volume_strategy.next()
        (b0, bf) = self.bval_strategy.next()

        #return self.img.get_data()[x0:xf, y0:yf, z0:zf, bi];
        return (x0, xf, y0, yf, z0, zf, b0, bf)

    def size(self):
        return self.volume_strategy.size()

class DmriCubicPatchVolumeRandomIndexer(DmriVolumeRandomIndexer):

    def __init__(self, img_shape, patch):
        vol_strategy = self._get_volume_strategy(img_shape, patch)
        super(DmriCubicPatchVolumeRandomIndexer, self).__init__(img_shape, vol_strategy)

    def _get_volume_strategy(self, img_shape, patch):
        #return Random3DRange(img_shape[0:3], patch, patch, patch)
        return All3DRangePosibleNotOverlapping(img_shape[0:3], patch, patch, patch)



class   DmriLrHrCubicPatchVolumeRandomIndexer(object):
    def __init__(self, img_lr_shape, n, m):
        self.img_lr_shape = img_lr_shape
        self.n = n;
        self.m = m
        self.patch = 2 * n + 1;
        self.dts = DmriCubicPatchVolumeRandomIndexer(img_lr_shape, self.patch)


    def size(self):
        #return min(self.img_lr_shape)-self.patch
        return self.dts.size()

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        (x0, xf, y0, yf, z0, zf, b0, bf) = self.dts.next()
        n = self.n
        m = self.m
        lr_patch_shape = (x0, xf, y0, yf, z0, zf, b0, bf)
        hr_patch_shape = ((x0+n)*m, (x0+n+1)*m, (y0+n)*m, (y0+n+1)*m,(z0+n)*m, (z0+n+1)*m, b0, bf)
        return {'lr': lr_patch_shape, 'hr':hr_patch_shape}


