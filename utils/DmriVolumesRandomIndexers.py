from utils.RandomRanges import Fixed1DRange, Random3DRange, All3DRangePosibleNotOverlapping



class DmriVolumeRandomIndexer(object):
    """
        Util para tomar volumnes random de una imagen 3D con varios b-vals
        Si no se pasa estrategia de seleccion de volumen se usa la estrategia
        randon 'Random3DRange'.
        Si no se pasa estrategia de seleccion de bvalores se usa la estrategia
        de rango fijo 'Fixed1DRange'

        =[COMPATIVILIDAD PARA ATRAS]:==========================================
        Equivalente llamarlo
            antes :
                DmriVolumeRandomIndexer(shape)
            ahora
                volume_strategy = Random3DRange(shape[0:3],shape[0]//patches, shape[1]//2, shape[2]//2)
                bval_strategy = Fixed1DRange(0,img_shape[3])
                DmriVolumeRandomIndexer(volume_strategy, bval_strategy)
        ========================================================================

        @see utils.RandomRanges.Random3DRange
        @see utils.RandomRanges.Fixed1DRange
    """
    def __init__(self, volume_strategy, bval_strategy):
        self.volume_strategy = volume_strategy
        self.bval_strategy = bval_strategy

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        (x0, xf, y0, yf, z0, zf) = self.volume_strategy.next()
        (b0, bf) = self.bval_strategy.next()
        return (x0, xf, y0, yf, z0, zf, b0, bf)

    def size(self):
        return min(self.volume_strategy.size(),self.bval_strategy.size())

class DmriCubicPatchVolumeRandomIndexer(DmriVolumeRandomIndexer):
    """
        DEPRECATED
        reemplazar por:
          antes:
            DmriCubicPatchVolumeRandomIndexer(img_shape, patch)
          ahora:
            vol_strategy = All3DRangePosibleNotOverlapping(img_shape[0:3], patch, patch, patch)
            bval_strategy = Fixed1DRange(0,img_shape[3])
            DmriVolumeRandomIndexer(vol_strategy, bval_strategy)

        @see  utils.RandomRanges.All3DRangePosibleNotOverlapping

    """
    def __init__(self, img_shape, patch):
        raise RuntimeError('Deprecated')


class DmriLrHrCubicPatchVolumeRandomIndexer(object):
    """
        Idem 'DmriVolumeRandomIndexer' pero devuelve los indices
        para una version Hr y Lr del volumen

        @param: fconvert funcion qque toma una tupla (6,) en hr y devuelve
                el equivalente en lr
                ej:

                   lambda (x0, xf, y0, yf, z0, zf, b0, bf) : ((x0+n)*m, (x0+n+1)*m, (y0+n)*m, (y0+n+1)*m,(z0+n)*m, (z0+n+1)*m, b0, bf)

    """
    def __init__(self, dmri_volume_indexer, fconvert, from_lr2hr=True):
        self.dmri_volume_indexer = dmri_volume_indexer
        self.fconvert = fconvert
        self._from_lr2hr = from_lr2hr

    def size(self):
        return self.dmri_volume_indexer.size()

    def __iter__(self):
        return self

    def next(self):
        if self._from_lr2hr:
            lr_patch_shape = self.dmri_volume_indexer.next()
            hr_patch_shape = self.fconvert(*lr_patch_shape)
        else:
            hr_patch_shape = self.dmri_volume_indexer.next()
            lr_patch_shape = self.fconvert(*hr_patch_shape)

        return {'lr': lr_patch_shape, 'hr':hr_patch_shape}

