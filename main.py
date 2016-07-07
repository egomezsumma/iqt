import dipy.reconst.dti as dti
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import time

from sklearn import datasets, linear_model

from utils.DataGetter import DataGetter
from utils.DownsampledImage import DownsampledImage
from utils.DmriSampleCreators import LrHrDmriRandomSampleCreator
from utils.dmri_patch_operations.LrHrPatchIterator import LrHrPatchIterator
from utils.dmri_patch_operations.DmriPatch import DmriPatch
from utils.dmri_patch_operations.DtiModel import DtiModel
from utils.img_utils import column_this, padding
from utils.ml.MLDataBuilder import SimpleDtiMlDataBuilder
from utils.persistance.MLPersistence import MLPersistence


## Mock methodsfrom utils.ml.MLDataBuilder import SimpleDtiMlDataBuilder
def get_a_patch_to_predict():
    return np.arange(5 * 5 * 5).reshape(5, 5, 5);

def load_dmri(n_samples, n , m):
    try:
        N1 = (2 * n + 1) ** 3
        N2 = m ** 3

        d = DataGetter()
        datas_names = [
            DataGetter.STANDFORD_HARDI_DATA,
            #DataGetter.TAIWAN_NTU_DSI_DATA,
            #DataGetter.SHERBROOKE_3SHELL_DATA
        ]

        datas = d.get_data(datas_names)

        lr_hr_imgs = [DownsampledImage(name, datas[name]['img'], datas[name]['gtab'], m) for name in datas_names]
        sample_creators = [LrHrDmriRandomSampleCreator(lr_hr_img, n, m) for lr_hr_img in lr_hr_imgs]
        sdb = SimpleDtiMlDataBuilder(sample_creators, n_samples);
        X, Y = sdb.build()
        print "X:", X.shape, "Y:", Y.shape
        return X, Y, lr_hr_imgs
    except Exception as e:
        print e;


def train(X, Y):
    # Split the data into training/testing sets
    dmri_X_train = X.T[:-1]
    dmri_X_test = X.T[-1:]

    # Split the targets into training/testing sets
    dmri_y_train = Y.T[:-1]
    dmri_y_test = Y.T[-1:]

    # Create linear regression object
    regr = linear_model.LinearRegression()


    # Train the model using the training sets
    print 'Training set_training.shape=', dmri_X_train.T.shape, ' ->target.shape=', dmri_y_train.T.shape
    try:
        regr.fit(dmri_X_train, dmri_y_train)
    except RuntimeError as e:
        print e;
    return regr, dmri_X_train , dmri_y_train, dmri_X_test, dmri_y_test



name_experiment='experimento1'
n, m = 2, 2
n_samples = 20


regr = MLPersistence.load(name_experiment)
if regr is None or True:
    X, Y, lr_hr_imgs = load_dmri(n_samples, n, m)

    ## Entreno con todas las samples menos la ultima que la uso para test
    regr, dmri_X_train , dmri_y_train, dmri_X_test, dmri_y_test = train(X,Y)

    ## Guardo en archivo
    MLPersistence.save(regr, name_experiment)

    ## Impresion de variables del fiteo
    # The mean square error
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(dmri_X_test) - dmri_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(dmri_X_test, dmri_y_test))

    ## Pruebita
    #y = regr.predict(dmri_X_test)
    #print dmri_y_test, y
    #print y.shape
    #voxels_dxx = y.reshape((6,8)).T
    #dti_vol = voxels_dxx.reshape((2,2,2,6))
    #print dti_vol.shape

## Tomo una imagen para reconstruir
img_a_reconstruir = lr_hr_imgs[0];

## Hacerle el pading
img_lr = padding(img_a_reconstruir.get_lr_img(), 2)


## Shape de las imagenes
img_hr_shape = img_a_reconstruir.get_hr_img().shape
img_lr_shape = img_lr.shape
print 'img_lr_shape', img_lr_shape

## Armo el esqueleto de la dti-imagen
# ???????????? no entiendo porque necesita 2*m mas ???????????????

sx, sy, sz, _ = img_lr_shape

dti_img_hr = np.zeros((sx*m, sy*m, sz*m, 6));

## Creo un iterador sobre la imagen lr
it = LrHrPatchIterator(img_lr_shape, n, m)

img_hr_gtab = img_a_reconstruir.get_gtab();
dti_model = DtiModel(img_hr_gtab);

sum=0;

img_lr_dti = dti_model._fit_model(img_lr)

start_time = time.time()
for data_ranges_lr_hr in it :
    sum=sum+1;

    # Indices en la imagen lr
    x0, xf, y0, yf, z0, zf = data_ranges_lr_hr['lr']

    # Fiteo el modelo del patch-lr (6 params dti)
    dti_patch_vol = img_lr_dti[x0:xf, y0: yf, z0:zf]

    x_vol = dti_patch_vol #dti_model.get_dti_params(dti_patch_vol)

    # Lo estructuro para que lo entienda ML (5,5,5,6) --> (1x750)
    x = column_this(x_vol)
    y = regr.predict(x.T)
    # Lo reestructuro (1,48) --> (2,2,2,6)
    dti_vol = y.reshape((2, 2, 2, 6))

#    if x_vol[0,0,0,:].any():
#        print dti_patch_vol.shape, xf, x0, yf, y0, zf, z0
#        print x_vol[0,0,0,:]
#        print x_vol.shape, x[:6]
#        print x.T
#        print y.shape
#        print y.T

    # Indices equivalentes en la imagen hr
    (a0, af, b0, bf, c0, cf) = data_ranges_lr_hr['hr']
    dti_img_hr[a0:af, b0:bf, c0:cf, :] = dti_vol

    #print np.sum(img_lr[x0:xf, y0: yf, z0:zf]), img_lr[x0:xf, y0: yf, z0:zf].shape
    #print np.sum(img_hr[x0:xf, y0: yf, z0:zf]), img_hr[x0:xf, y0: yf, z0:zf].shape
    # if zf >= img_lr_shape[2] - 1:
    #    print sum, (x0, xf, y0, yf, z0, zf), img_lr[x0:xf, y0: yf, z0:zf, 0].size


seg = time.time() - start_time
min = int(seg / 60)
print("--- time of predictions patchs : %d' %d'' --- num. iterations: %d" % (min , seg%60, sum))


# La corto porque tenia dimensiones de longitud impares
dti_img_hr = dti_img_hr[0:81,0:106,0:76, :]
print 'dti_img_hr=', dti_img_hr.shape


start_time = time.time()

## Agarrar el fiteo y pasarlo a imagen
tenmodel = dti.TensorModel(img_a_reconstruir.get_gtab())
#tensors = dti.from_lower_triangular(dti_img_hr)
# (12 dti params --eval--evecs--)
tensors = dti.eig_from_lo_tri(dti_img_hr)
print 'tensors', tensors.shape
img_reconstructed = tenmodel.predict(tensors)
print 'img_reconstructed', img_reconstructed.shape


seg = time.time() - start_time
min = int(seg / 60)
print("--- time of reconstruction : %d' %d'' --- num. iterations: %d" % (min, seg%60, sum))

# guardar el resultado
nifti1img = nib.Nifti1Image(img_reconstructed, img_a_reconstruir.get_hr_affine())
nib.save(nifti1img, name_experiment + '.nii.gz')
print 'img_hr_shape=', img_hr_shape, 'img_lr_shape=', img_lr_shape




# ???? ESTA VALIDACION NO ME DA #######################
#print sum, sum*8, (img_hr_shape[0]-5)*(img_hr_shape[1]-5)*(img_hr_shape[2]-5)
#print img_hr_shape[0]-8,img_hr_shape[1]-8,img_hr_shape[2]-8


#TODO: chequear que escribia tantos voxeles como se supone (acordate de restarle el pading)
#      creo q seria 2*2*2*(numit) == (x-2)*m * (y-2)*m * (z-2)*m (x,y,z)=lr.shape