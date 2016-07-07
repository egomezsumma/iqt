from sklearn.externals import joblib


class MLPersistence(object):

    @staticmethod
    def save(model, name):
        joblib.dump(model, name + '.pkl')

    @staticmethod
    def load(name):
        try:
            return joblib.load(name + '.pkl')
        except IOError:
            return None
