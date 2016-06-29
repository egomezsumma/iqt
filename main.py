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


# input:
#      img:{get_data,get_affine, get_header:{get_zooms,...} }
#      scale: R
# output: new_data, new_affine
#      if scale=2 --> la mitad
#      if scale=0.5 --> el doble
def downsampling(img, scale):
    from dipy.align.reslice import reslice
    data = img.get_data()
    affine = img.get_affine()
    # Load and show the zooms which hold the voxel size.
    zooms = img.get_header().get_zooms()[:3]
    data2, affine2 = reslice(data, affine, zooms, 2 * np.array(zooms))
    return data2, affine2

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

print xis.shape;

d = DataGetter()


print d.get_data().keys();