from dipy.data import fetch_stanford_hardi, read_stanford_hardi
import numpy as np
from utils.DataGetter import DataGetter


## MOCKS METHODS #######################################################################################################

def get_trainig_set():
    xi = []
    yi = []
    for i in range(0, 3):
        xi.append(np.arange(5 * 5 * 5).reshape(5, 5, 5))
        yi.append(np.arange(2 * 2 * 2).reshape(2, 2, 2))
    return (np.array(xi), np.array(yi));


def get_a_patch_to_predict():
    return np.arange(5 * 5 * 5).reshape(5, 5, 5);


## MATH METHODS ########################################################################################################

# Dada una matriz de NxM y la convierte en una
# de Kx1 (con k = N*M)
def column_this(matrix):
    res = matrix2vector(matrix)
    return np.array([res]).T


# Dada una matriz de NxM y la convierte en una
# de Nx(M+1)
# new_column  tiene que ser de PxQ con P*Q=N
def append_column(matrix, new_column):
    b1 = matrix2vector(new_column)
    return np.append(matrix, np.array([b1]).T, axis=1)


# Dada una matriz de NxM la aplana en una de Px1
# con P=N*M
def matrix2vector(matrix):
    return np.reshape(matrix, matrix.size);


def psudo_inversa(matriz):
    return matriz



# Set parameters
n=2
m=3
p2=p1=6
N1=(2*n+1)**3
N2=m**3

n, m, p1, p2, N1, N2

#mock que toma el training set
(xis, yis) = get_trainig_set()

#gl = GlobalLinearRegression(xis, yis, [])
#gl.train();

#y, G = gl.predict(x);

#print xis.shape;

from utils.DmriVolumesRandomIndexers import DmriLrHrCubicPatchVolumeRandomIndexer
from utils.DmriSampleCreators import LrHrDmriRandomSampleCreator


d = DataGetter()
data = d.get_data(DataGetter.STANDFORD_HARDI_DATA)['standfor_hardi'];
try:
    n=2;m=2

    sc = LrHrDmriRandomSampleCreator(data['img'], n, m)

    arr = [ sc.next() for _ in range(0,6)]
    print arr
except Exception as e:
    print e;

print 'alala'