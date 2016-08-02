from utils.RandomRanges import Fixed1DRange, Random3DRange, All3DRangePosibleNotOverlapping
import math



class DmriVolumeRandomIndexer(object):
    """
        Util para tomar volumnes random de una imagen 3D con varios b-vals
        Si no se pasa estrategia de seleccion de volumen se usa la estrategia
        randon 'Random3DRange'.
        Si no se pasa estrategia de seleccion de bvalores se usa la estrategia
        de rango fijo 'Fixed1DRange'

        @see utils.RandomRanges.Random3DRange
        @see utils.RandomRanges.Fixed1DRange
    """
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
    """
        Idem DmriVolumeRandomIndexer pero que utiliza la estrategia 'All3DRangePosibleNotOverlapping'
        para seleccionar el volumne
        @see  utils.RandomRanges.All3DRangePosibleNotOverlapping

    """
    def __init__(self, img_shape, patch):
        vol_strategy = self._get_volume_strategy(img_shape, patch)
        super(DmriCubicPatchVolumeRandomIndexer, self).__init__(img_shape, vol_strategy)

    def _get_volume_strategy(self, img_shape, patch):
        #return Random3DRange(img_shape[0:3], patch, patch, patch)
        return All3DRangePosibleNotOverlapping(img_shape[0:3], patch, patch, patch)


class FixedDmriCubicPatchVolumeIndexer(object):
    def __init__(self, volume_range, bval_range, limit=-1):
        self.volume_range = volume_range
        self.bval_range = bval_range
        self.limit = limit
        self._current = 0

    def __iter__(self):
        return self

    def next(self):
        if self.limit > -1 and self._current >= self.limit:
            raise StopIteration
        else:
            self._current += 1
            (x0, xf, y0, yf, z0, zf) = self.volume_range
            (b0, bf) = self.bval_range
            return (x0, xf, y0, yf, z0, zf, b0, bf)

    def size(self):
        return self.limit

class DmriLrHrCubicPatchVolumeRandomIndexer(object):
    """
        Clase que dada una DownsampledImage devuelve rangos
        de sub volumenes de la misma en su version original (hr)
        y su eqiuvalente downsampleada (lr)
    """
    def __init__(self, img_lr_shape, n, m, dmri_volume_indexer=None):
        self.img_lr_shape = img_lr_shape
        self.n = n;
        self.m = m
        self.patch = 2 * n + 1;
        self.dts = dmri_volume_indexer
        self._img_lr_shape = img_lr_shape
        if dmri_volume_indexer is None:
            self._set_default_dmri_volume_indexer()

    def _set_default_dmri_volume_indexer(self):
        self.dts = DmriCubicPatchVolumeRandomIndexer(self._img_lr_shape, self.patch)

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

