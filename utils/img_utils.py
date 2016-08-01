import numpy as np;
import matplotlib.pyplot as plt


# input:
#      img:{get_data,get_affine, get_header:{get_zooms,...} }
#      scale: R
# output: new_data, new_affine
#      if scale=2 --> la mitad
#      if scale=0.5 --> el doble
def downsampling(img, scale=2):
    from dipy.align.reslice import reslice
    import numpy as np
    data = img.get_data()

    # hago que sean pares
    for axis, dim in enumerate(data.shape[:3]):
        if dim % 2 != 0:
            data = np.insert(data, dim, 0, axis=axis)

    affine = img.get_affine()
    #Load and show the zooms which hold the voxel size.
    zooms = img.get_header().get_zooms()[:3]
    data2, affine2 = reslice(data, affine, zooms, scale*np.array(zooms))
    return data2, affine2


def downsampling2(img, scale=2):
    from dipy.align.reslice import reslice
    import numpy as np
    data = img

    # hago que sean pares
    for axis, dim in enumerate(data.shape[:3]):
        if dim % 2 != 0:
            data = np.insert(data, dim, 0, axis=axis)

    affine = np.identity(4,dtype=float)
    #Load and show the zooms which hold the voxel size.
    zooms = np.array([1, 1, 1])
    dataD, affine2 = reslice(data, affine, zooms, np.dot(scale,np.array(zooms)))
    return dataD



# Dada una matriz de NxM y la convierte en una
# de Kx1 (con k = N*M)
def __column_this(matrix):
    res = matrix2vector(matrix)
    return np.array([res], dtype='float').T


# Dada una matriz de NxM y la convierte en una
# de Nx(M+1)
# new_column  tiene que ser de PxQ con P*Q=N
def __append_column(matrix, new_column):
    b1 = matrix2vector(new_column)
    return np.append(matrix, np.array([b1], dtype='float').T, axis=1)


# Dada una matriz de NxM la aplana en una de Px1
# con P=N*M
def __matrix2vector(matrix):
    return np.reshape(matrix, matrix.size);

# Dada una matriz de NxM y la convierte en una
# de Kx1 (con k = N*M)
def column_this(matrix):
    #return matrix.reshape(-1).T
    return np.array([matrix.reshape(-1)], dtype='float').T


# Dada una matriz de NxM y la convierte en una
# de Nx(M+1)
# new_column  tiene que ser de PxQ con P*Q=N
def append_column(matrix, new_column):
    b1 = matrix2vector(new_column)
    return np.append(matrix, np.array([b1], dtype='float').T, axis=1)


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

def padding(matrix, pading_size, value=0):
    pad_dim = ((pading_size,pading_size),(pading_size,pading_size), (pading_size,pading_size), (0,0))
    res = np.lib.pad(matrix, pad_dim, 'constant', constant_values=(value,))
    return res


"""
def padding(matrix, pading_size, value=0):
    pad_dim = ((pading_size,pading_size),(pading_size,pading_size), (pading_size,pading_size), (0,0))
    res = np.lib.pad(matrix, pad_dim, 'constant', constant_values=(value,))
    return res
"""

def _is(volumen, y=2, b=0, inter='none', cmap='gray'):
    plt.imshow(np.rot90(volumen[:, volumen.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    plt.axis('off')
    plt.colorbar()
    return plt



def _is(volumen, y=2, b=0, inter='none', cmap='gray'):
    if len(volumen.shape) > 3 :
        plt.imshow(np.rot90(volumen[:, volumen.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    else:
        plt.imshow(np.rot90(volumen[:, volumen.shape[1] // y]), interpolation=inter, cmap=cmap)
    plt.axis('off')
    plt.colorbar()
    return plt


def _is3d(volumen, y=2, b=0, inter='none', cmap='gray'):
    plt.imshow(np.rot90(volumen[:, volumen.shape[1] // y, :]), interpolation=inter, cmap=cmap)
    plt.axis('off')
    plt.colorbar()
    return plt


def _isc(vol1, vol2, y=2, b=0, inter='none', cmap='gray'):
    # plt.figure('Showing the datasets')
    plt.subplot(1, 2, 1).set_axis_off()
    plt.imshow(np.rot90(vol1[:, vol1.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    plt.subplot(1, 2, 2).set_axis_off()
    plt.imshow(np.rot90(vol2[:, vol2.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    #plt.colorbar()
    return plt

def _isc3(vol1, vol2, vol3, y=2, b=0, inter='none', cmap='gray'):
    # plt.figure('Showing the datasets')
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(np.rot90(vol1[:, vol1.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(np.rot90(vol2[:, vol2.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(np.rot90(vol3[:, vol3.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    plt.show()
    plt.colorbar()
    return plt


def buildDownsampligBy2(nx, ny, nz):
    """
    Build the downsamplig matrix for reduce 50% a 3d image
    use:
     D = buildDownsampligBy2(A.shape)
     small = D.dot(A.reshape(-1)).reshape(A.shape/2)
    :param nx, ny, nz:
    :return: sparce matrix (nx*ny*nz/8, nx*ny*nz)
    """
    from scipy.sparse import lil_matrix
    new_nx = nx / 2
    new_ny = ny / 2
    new_nz = nz / 2
    D = lil_matrix((new_nx * new_ny * new_nz, nx * ny * nz,))
    avg = 0.125

    conv_pos = lambda i, j, k: i * ny * nz + j * nz + k
    conv_pos_new = lambda i, j, k: i * new_ny * new_nz + j * new_nz + k

    for i in range(0, new_nx):
        for j in range(0, new_ny):
            for k in range(0, new_nz):
                row = conv_pos_new(i, j, k)
                ii, jj, kk = 2 * i, 2 * j, 2 * k
                row00 = conv_pos(ii, jj, kk)
                row01 = conv_pos(ii, jj + 1, kk)
                row10 = conv_pos(ii + 1, jj, kk)
                row11 = conv_pos(i + 1, jj + 1, kk)
                D[row, row00:row00 + 2] = avg
                D[row, row01:row01 + 2] = avg
                D[row, row10:row10 + 2] = avg
                D[row, row11:row11 + 2] = avg
    return D



def buildDownsamplig(nx, ny, nz, factor):
    """
    Build the downsamplig matrix for reduce (100/factor)% a 3d image
    use:
     D = buildDownsampligBy2(A.shape , factor)
     small = D.dot(A.reshape(-1)).reshape(A.shape/factor**3)
    :param nx, ny, nz:
    :return: sparce matrix (nx*ny*nz/factor**3, nx*ny*nz)
    """
    from scipy.sparse import lil_matrix
    new_nx = nx / factor
    new_ny = ny / factor
    new_nz = nz / factor
    D = lil_matrix((new_nx * new_ny * new_nz, nx * ny * nz))
    avg = 1.0/(factor**3)

    conv_pos = lambda i, j, k: i * ny * nz + j * nz + k
    conv_pos_new = lambda i, j, k: i * new_ny * new_nz + j * new_nz + k

    for i in range(0, new_nx):
        for j in range(0, new_ny):
            for k in range(0, new_nz):
                row = conv_pos_new(i, j, k)
                ii, jj, kk = factor * i, factor * j, factor * k
                for u in range(0,factor):
                    for v in range(0, factor):
                        rowuv = conv_pos(ii+u, jj+v, kk)
                        D[row, rowuv:rowuv + factor] = avg
    return D
