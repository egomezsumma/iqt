import numpy as np
import nibabel as nib

## Mock methodsfrom utils.ml.MLDataBuilder import SimpleDtiMlDataBuilder
def get_a_patch_to_predict():
    return np.arange(5 * 5 * 5).reshape(5, 5, 5);

def print_info(img_recons, name):
    print name, ': (min:max)', np.min(img_recons), ':',np.max(img_recons), 'ptp:', np.ptp(img_recons)
    print '#inf: ', np.sum(np.isposinf(img_recons)), '#-inf: ', np.sum(
        np.isneginf(img_recons)), ' #Nan: ', np.sum(np.isnan(img_recons))
    print '#uniques=', np.unique(img_recons).size
    cantidad, valor = np.histogram(img_recons.reshape(-1))
    print 'histograma:'
    print '    #voxels: ', cantidad
    print '    valor: ', valor

    print 'dtype', img_recons.dtype
    print

def save_nifti(name, data, affine):
    nifti1img = nib.Nifti1Image(data, affine)
    nib.save(nifti1img, name+ '.nii.gz')


def test():
    print 'test ok'