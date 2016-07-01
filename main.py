import numpy as np
from utils.DataGetter import DataGetter
from utils.ml.MLDataBuilder import SimpleDtiMlDataBuilder


## MOCKS METHODS #######################################################################################################

def get_a_patch_to_predict():
    return np.arange(5 * 5 * 5).reshape(5, 5, 5);




from utils.DmriSampleCreators import LrHrDmriRandomSampleCreator

n = 2; m = 2;
N1=(2*n+1)**3
N2=m**3

d = DataGetter()
datas_names = [
            DataGetter.STANDFORD_HARDI_DATA,
            DataGetter.TAIWAN_NTU_DSI_DATA
        ];

datas = d.get_data(datas_names);


try:

    sample_creators = [ LrHrDmriRandomSampleCreator(name, datas[name]['img'], datas[name]['gtab'], n, m) for name in datas_names ]
    sdb = SimpleDtiMlDataBuilder(sample_creators,6);
    X, Y = sdb.build()
    print X.shape, Y.shape


    """
    print "Samples for standfor_hardi:", sc.size();
    arr = [ sc.next() for _ in range(0,10)]

    dtim = DtiModel(data['gtab'])
    patch_lr, patch_hr = arr[0]

    x_dti_patch = dtim.get_dti_params(patch_lr);
    y_dti_patch = dtim.get_dti_params(patch_hr);


    X = column_this(x_dti_patch.get_volume())
    Y = column_this(y_dti_patch.get_volume())

    print X.shape, Y.shape
    for i in range(1, len(arr)):
        patch_lr, patch_hr = arr[i]
        x_dti_patch = dtim.get_dti_params(patch_lr);
        y_dti_patch = dtim.get_dti_params(patch_hr);

        X = append_column(X, x_dti_patch.get_volume())
        Y = append_column(Y, y_dti_patch.get_volume())

    print X.shape, Y.shape
    """
except Exception as e:
    print e;

print 'alala'