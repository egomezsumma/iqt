import numpy as np;
import matplotlib.pyplot as plt

# input:
#      img:{get_data,get_affine, get_header:{get_zooms,...} }
#      scale: R
# output: new_data, new_affine
#      if scale=2 --> la mitad
#      if scale=0.5 --> el doble
def downsampling(img, scale):
    from dipy.align.reslice import reslice
    import numpy as np
    data = img.get_data()
    affine = img.get_affine()
    #Load and show the zooms which hold the voxel size.
    zooms = img.get_header().get_zooms()[:3]
    data2, affine2 = reslice(data, affine, zooms, 2*np.array(zooms))
    return data2, affine2


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

def plot_this(x,y, model):
    plt.scatter(x, y,  color='black')
    print x.shape, model.predict(x).shape
    plt.plot(x, model.predict(x), color='blue',
             linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
