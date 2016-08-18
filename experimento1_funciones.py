import numpy as np
import nibabel as nib
from utils.DataGetter import DataGetter
from utils.DownsampledImage import DownsampledImage
from utils.DmriSampleCreators import LrHrDmriRandomSampleCreator
from utils.ml.MLDataBuilder import SimpleDtiMlDataBuilder
#from scipy.sparse import csr_matrix, csc_matrix


from sklearn import linear_model

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


def load_dmri(n_samples, n , m):
    try:
        N1 = (2 * n + 1) ** 3
        N2 = m ** 3

        d = DataGetter()
        datas_names = [
            DataGetter.STANDFORD_HARDI_DATA, #(81, 106, 76, 160)
            #DataGetter.TAIWAN_NTU_DSI_DATA,  #(96, 96, 60, 203)
            #DataGetter.SHERBROOKE_3SHELL_DATA #(128, 128, 60, 193)
        ]

        #    ksaknlaknska;
        #    import numpy.ma as ma
        #    x = np.array([1, 2, 3, -1, 5])
        #    mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])

        datas = d.get_data(datas_names)

        lr_hr_imgs = [DownsampledImage(name, datas[name]['img'], datas[name]['gtab'], m) for name in datas_names]
        print [(name, datas[name]['img'].shape) for name in datas_names]
        sample_creators = [LrHrDmriRandomSampleCreator(lr_hr_img, n, m) for lr_hr_img in lr_hr_imgs]
        sdb = SimpleDtiMlDataBuilder(sample_creators, n_samples);
        X, Y = sdb.build()
        print "X:", X.shape, "Y:", Y.shape
        return X, Y, lr_hr_imgs
    except Exception as e:
        print e;


def train(X, Y, verbose=False):
    # Split the data into training/testing sets
    dmri_X_train = X.T[:-1]
    dmri_X_test = X.T[-1:]

    # Split the targets into training/testing sets
    dmri_y_train = Y.T[:-1]
    dmri_y_test = Y.T[-1:]

    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=False)
    #regr = linear_model.Ridge(alpha = .5)


    # Train the model using the training sets
    if verbose :
        print 'Training set_training.shape=', dmri_X_train.T.shape, ' ->target.shape=', dmri_y_train.T.shape

    try:
        regr.fit(dmri_X_train, dmri_y_train)
    except RuntimeError as e:
        print e;
    return regr, dmri_X_train , dmri_y_train, dmri_X_test, dmri_y_test