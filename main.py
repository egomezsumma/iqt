import numpy as np
from utils.DataGetter import DataGetter
from utils.ml.MLDataBuilder import SimpleDtiMlDataBuilder
from utils.DownsampledImage import DownsampledImage
from utils.img_utils import plot_this

## MOCKS METHODS #######################################################################################################

def get_a_patch_to_predict():
    return np.arange(5 * 5 * 5).reshape(5, 5, 5);


from utils.DmriSampleCreators import LrHrDmriRandomSampleCreator

def load_dmri(n_samples, n , m):
    try:
        N1 = (2 * n + 1) ** 3
        N2 = m ** 3

        d = DataGetter()
        datas_names = [
            DataGetter.STANDFORD_HARDI_DATA,
            DataGetter.TAIWAN_NTU_DSI_DATA
        ];

        datas = d.get_data(datas_names);


        lr_hr_imgs = [DownsampledImage(name, datas[name]['img'], datas[name]['gtab'], m) for name in datas_names ];
        sample_creators = [ LrHrDmriRandomSampleCreator(lr_hr_img, n, m) for lr_hr_img in lr_hr_imgs ]
        sdb = SimpleDtiMlDataBuilder(sample_creators, n_samples);
        X, Y = sdb.build()
        print X.shape, Y.shape
        return X, Y, lr_hr_imgs
    except Exception as e:
        print e;


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

n = 2;
m = 2;

X, Y , lr_hr_imgs = load_dmri(10, n, m)

#Split the data into training/testing sets
dmri_X_train = X.T[:-1]
dmri_X_test = X.T[-1:]

# Split the targets into training/testing sets
dmri_y_train = Y.T[:-1]
dmri_y_test = Y.T[-1:]


# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
print 'Training set_training.shape=', dmri_X_train.T.shape , ' ->target.shape=', dmri_y_train.T.shape

try:
    regr.fit(dmri_X_train, dmri_y_train)
except RuntimeError as e:
    print e;


# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(dmri_X_test) - dmri_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(dmri_X_test, dmri_y_test))


y = regr.predict(dmri_X_test)
print y.shape
voxels_dxx = y.reshape((6,8)).T
dti_vol = voxels_dxx.reshape((2,2,2,6))
print dti_vol.shape



from utils.img_utils import column_this
from utils.dmri_patch_operations.LrHrPatchIterator import LrHrPatchIterator
from utils.dmri_patch_operations.DmriPatch import DmriPatch
from utils.dmri_patch_operations.DtiModel import DtiModel

img_a_reconstruir = lr_hr_imgs[0];
img_lr_gtab = img_a_reconstruir.get_gtab();
img_lr = img_a_reconstruir.get_lr_img()
img_hr_shape = img_a_reconstruir.get_hr_img().shape
dti_img_hr = np.zeros((img_hr_shape[0], img_hr_shape[1], img_hr_shape[2], 6));
print dti_img_hr.shape
img_lr_shape = img_lr.shape

it = LrHrPatchIterator(img_lr_shape,n, m)



dti_model = DtiModel(img_lr_gtab);
for data_ranges_lr_hr in it :
    (x0, xf, y0, yf, z0, zf) = data_ranges_lr_hr['lr']

    dmri_pathc_data = img_lr[x0:xf, y0: yf, z0:zf]
    x = dti_model.get_dti_params(DmriPatch(dmri_pathc_data)).get_volume()
    #print x.shape
    x = column_this(x)
    y = regr.predict(x.T)
    #print y.shape

    voxels_dxx = y.reshape((6, 8)).T
    dti_vol = voxels_dxx.reshape((2, 2, 2, 6))

    (a0, af, b0, bf, c0, cf) = data_ranges_lr_hr['hr']
    dti_img_hr[a0:af, b0:bf, c0:cf, :] = dti_vol

    #print np.sum(img_lr[x0:xf, y0: yf, z0:zf]), img_lr[x0:xf, y0: yf, z0:zf].shape
    #print np.sum(img_hr[x0:xf, y0: yf, z0:zf]), img_hr[x0:xf, y0: yf, z0:zf].shape

print dti_img_hr.shape

#TODO: chequear que escribia tantos voxeles como se supone (acordate de restarle el pading)
#      creo q seria 2*2*2*(numit) == (x-2)*m * (y-2)*m * (z-2)*m (x,y,z)=lr.shape