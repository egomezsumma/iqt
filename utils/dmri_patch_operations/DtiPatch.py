
class DtiPatch(object):
    def __init__(self, dti_params_volume, indexs):
        self.volume = dti_params_volume;
        self.indexs = indexs

    def get_volume(self):
        return self.volume

    def get_indexs(self):
        return self.indexs
