from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel
from dipy.data import read_stanford_labels


class DataGetter(object):
    STANDFORD_HARDI_DATA = 'standfor_hardi';
    SHERBROOKE_3SHELL_DATA = 'sherbrooke_3shell';
    ANISO_VOX_DATA = 'aniso_vox';
    TAIWAN_NTU_DSI_DATA = 'taiwan_ntu_dsi';
    ISBI2013_2SHELL_DATA = 'isbi2013_2shell';
    STANDFORD_HARDI_LABELS = 'standfor_hardi_labels'

    DATAS_NAMES = [
        STANDFORD_HARDI_DATA,
        SHERBROOKE_3SHELL_DATA,
        ANISO_VOX_DATA,
        TAIWAN_NTU_DSI_DATA,
        ISBI2013_2SHELL_DATA,
        STANDFORD_HARDI_LABELS
    ]

    def __init__(self):
        self._cache = {};


    def get_data(self, name_s=None):
        if name_s is None:
            names = DataGetter.DATAS_NAMES
        else:
            names = name_s if hasattr(name_s, '__iter__') else [name_s];

        res = {}
        for name in names :
            if name in DataGetter.DATAS_NAMES :
                if name not in  self._cache.keys():
                    self._cache[name] = getattr(self, '_get_' + name)();
                res[name] = self._cache[name]
        return res;

    def _get_standfor_hardi_labels(self):
        hardi_img, gtab, labels_img = read_stanford_labels()
        return {'img': hardi_img, 'gtab':gtab, 'labels':labels_img}


    def _get_standfor_hardi(self):
        fetch_stanford_hardi()
        stanford_hardi_img, stanford_hardi_gtab = read_stanford_hardi()
        return {'img': stanford_hardi_img, 'gtab':stanford_hardi_gtab}
        #stanford_hardi_data = stanford_hardi_img.get_data()
        #stanford_hardi_affine = stanford_hardi_img.affine
        #print('Computing brain mask...')
        #b0_mask, mask = median_otsu(stanford_hardi_data)
        #print('Computing tensors...')
        #tenmodel = TensorModel(stanford_hardi_gtab)
        #tensorfit = tenmodel.fit(stanford_hardi_data, mask=mask)

    def _get_sherbrooke_3shell(self):
        # SHERBROOK #######################################################
        from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell
        fetch_sherbrooke_3shell()
        img, gtab = read_sherbrooke_3shell()
        #data = img.get_data()
        return {'img': img, 'gtab': gtab}

    def _get_aniso_vox(self):
        import nibabel as nib
        #from dipy.align.reslice import reslice
        from dipy.data import get_data

        fimg = get_data('aniso_vox')
        aniso_vox_img = nib.load(fimg)
        aniso_vox_data = aniso_vox_img.get_data()

        #print aniso_vox_data.shape
        #aniso_vox_affine = aniso_vox_img.get_affine()
        return {'img': aniso_vox_img, 'gtab': aniso_vox_data}

    def _get_taiwan_ntu_dsi(self):
        # TAIWAN NTU (96,96,60,203) #######################################################
        from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
        # from dipy.data import get_data, dsi_voxels

        fetch_taiwan_ntu_dsi();
        img, gtab = read_taiwan_ntu_dsi();
        #data = img.get_data()
        #affine = img.get_affine()
        return {'img': img, 'gtab': gtab}

    def _get_isbi2013_2shell(self):
        # ISBI2013 2SHELL (50, 50, 50, 64) #######################################################
        from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell

        fetch_isbi2013_2shell()
        img, gtab = read_isbi2013_2shell()
        #data = img.get_data()
        return {'img': img, 'gtab': gtab}

