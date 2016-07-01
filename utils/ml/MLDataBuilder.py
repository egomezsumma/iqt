import numpy as np;
from utils.dmri_patch_operations.DtiModel import DtiModel
from utils.img_utils import column_this, append_column


class SimpleDtiMlDataBuilder:
    #
    # weights = {nombre_imagen: n} --> n < 1 que porcentaje del total_sample_wished seran de esa imagen
    def __init__(self, sample_creator_s, total_sample_wished, weights=None):
        self._sum_weights = 0
        self._total_samples = total_sample_wished
        self._sample_creators = sample_creator_s if type(sample_creator_s) is list else [sample_creator_s];

        self._set_weights(self._sample_creators, weights);


    def build(self, X=None, Y=None):
        for sc in self._sample_creators:
            X, Y = self._create_sample_for(sc, self._get_total_samples_of(sc), X, Y)
        return X, Y

    def _create_sample_for(self, sample_creator, limit, X=None, Y=None):
        print "Samples for ", sample_creator.name, ":" , limit, 'of', sample_creator.size();
        arr = [sample_creator.next() for _ in range(0, limit)]

        dtim = DtiModel(sample_creator.get_gtab())
        patch_lr, patch_hr = arr[0]

        x_dti_patch = dtim.get_dti_params(patch_lr);
        y_dti_patch = dtim.get_dti_params(patch_hr);

        first_i = 0
        if X is None:
            first_i = 1
            X = column_this(x_dti_patch.get_volume())
            Y = column_this(y_dti_patch.get_volume())

        #print X.shape, Y.shape
        for i in range(first_i, len(arr)):
            patch_lr, patch_hr = arr[i]
            x_dti_patch = dtim.get_dti_params(patch_lr);
            y_dti_patch = dtim.get_dti_params(patch_hr);

            X = append_column(X, x_dti_patch.get_volume())
            Y = append_column(Y, y_dti_patch.get_volume())

        return X, Y

    def _set_weights(self, sample_creators, weights):
        self._weights = {};
        #weights if weights is None else {};

        for sc in sample_creators:
            self._weights[sc.name] = 1;
            if weights is not None and sc.name in weights :
                self._weights[sc.name] = weights[sc.name]

        self._sum_weights = np.sum(self._weights.values());

    def _get_total_samples_of(self, sc):
        return int(self._total_samples*self._weights[sc.name]/self._sum_weights);