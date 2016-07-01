import numpy as np
from utils.DataGetter import DataGetter
from utils.ml.MLDataBuilder import SimpleDtiMlDataBuilder
from utils.img_utils import plot_this

## MOCKS METHODS #######################################################################################################

def get_a_patch_to_predict():
    return np.arange(5 * 5 * 5).reshape(5, 5, 5);


from utils.DmriSampleCreators import LrHrDmriRandomSampleCreator

def load_dmri(n_samples):
    try:
        n = 2;
        m = 2;
        N1 = (2 * n + 1) ** 3
        N2 = m ** 3

        d = DataGetter()
        datas_names = [
            DataGetter.STANDFORD_HARDI_DATA,
            DataGetter.TAIWAN_NTU_DSI_DATA
        ];

        datas = d.get_data(datas_names);

        sample_creators = [ LrHrDmriRandomSampleCreator(name, datas[name]['img'], datas[name]['gtab'], n, m) for name in datas_names ]
        sdb = SimpleDtiMlDataBuilder(sample_creators,n_samples);
        X, Y = sdb.build()
        print X.shape, Y.shape
        return X, Y
    except Exception as e:
        print e;


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


X, Y = load_dmri(10)

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

voxels_dxx = y.reshape((6,8)).T
dti_vol = voxels_dxx.reshape((2,2,2,6))
from utils.dmri_patch_operations.DtiModel import DtiPatch


DtiPatch()
print np.linalg.norm(y-dmri_y_test)



#plot_this(dmri_X_test,dmri_y_test,regr)



