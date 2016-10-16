import numpy as np;
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# input:
#      img:{get_data,get_affine, get_header:{get_zooms,...} }
#      scale: R
# output: new_data, new_affine
#      if scale=2 --> la mitad
#      if scale=0.5 --> el doble
def downsampling(img, scale=2):
    from dipy.align.reslice import reslice
    import numpy as np

    data = img
    if hasattr(img, 'get_data') :
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
    zooms = np.array([1.0, 1.0, 1.0])
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
    return np.reshape(matrix, matrix.size, order='F');

# Dada una matriz de NxM y la convierte en una
# de Kx1 (con k = N*M)
def column_this(matrix):
    #return matrix.reshape(-1).T
    return np.array([matrix.reshape(-1, order='F')], dtype='float').T


# Dada una matriz de NxM y la convierte en una
# de Nx(M+1)
# new_column  tiene que ser de PxQ con P*Q=N
def append_column(matrix, new_column):
    b1 = matrix2vector(new_column)
    return np.append(matrix, np.array([b1], dtype='float').T, axis=1)


# Dada una matriz de NxM la aplana en una de Px1
# con P=N*M
def matrix2vector(matrix):
    return np.reshape(matrix, matrix.size, order='F');

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

def _is(volumen, y=-1, b=0, inter='none', cmap='gray', title=None,vmin=None, vmax=None):
    if y <= 0 :
        y = volumen.shape[1] // 2

    plt.imshow(np.rot90(volumen[:,y, :, b]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title is not None:
       plt.title(title)
    #plt.colorbar()
    return plt





def _is(volumen, y=2, b=0, inter='none', cmap='gray', title=None,vmin=0, vmax=500):
    if len(volumen.shape) > 3 :
        plt.imshow(np.rot90(volumen[:, volumen.shape[1] // y, :, b]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(np.rot90(volumen[:, volumen.shape[1] // y]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    #plt.colorbar()
    #print volumen.shape, 'y=', volumen.shape[1]//y

    return plt

def _ig(volumen,fname, y=2, b=0, inter='none', cmap='gray',title=None,vmin=None, vmax=None):
    if title is not None:
        plt.title(title)

    if len(volumen.shape) > 3 :
        plt.imsave(fname, np.rot90(volumen[:, volumen.shape[1] // y, :, b]), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imsave(fname,np.rot90(volumen[:, volumen.shape[1] // y]), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')

    #plt.colorbar()
    #print volumen.shape, 'y=', volumen.shape[1]//y

    return plt




def _ish(volumen, y=-1, b=0, inter='none', cmap='gray', title=None, vmin=0, vmax=500):
    if y < 0 :
        y = volumen.shape[1] // 2

    plt.subplot(2, 2, 1).set_axis_off()
    if len(volumen.shape) > 3 :
        plt.imshow(np.rot90(volumen[:,  y, :, b]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(np.rot90(volumen[:, y]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    #plt.axis('off')

    plt.subplot(2, 1, 2).set_axis_off()
    plt.hist(volumen[:,y, :, b].flatten())
    plt.axis('on')

    u, std = volumen[:,y, :, b].mean(), np.std(volumen[:,y, :, b])
    if title is not None:
        plt.subplot(2, 2, 1).title(title)
    plt.subplot(2, 1, 2).set_title('Histogram y='+str(y)+' b='+str(b) +' u='+str(int(u))+'+/-'+str(int(std)))

    #plt.colorbar()
    #print volumen.shape, 'y=', volumen.shape[1]//y
    return plt



def _iswr(volumen, rect, y=2, b=0, inter='none', cmap='gray', title=None,linewidth=2, edgecolor='r',vmin=0, vmax=500):
    """
    :param volumen: Volumen o imagen a mostrar
    :param rect: tupe (orig_x, orig_y, width, height)
    :param y: En donde dividir el volumen, defalut 2 -> es decir se muestra la feta de la mitad en dimension 'y'
    :param b: Para volumenes con mas de 4 dimensiones. La seleccion de la cuarta dimension (ejemplo que b-valor en dwi)
    :param inter: Parametro de imshow
    :param cmap: Parametro de imshow
    :param title: Titulo de la imagen
    :return:
    """
    fig, ax = plt.subplots(1)
    if len(volumen.shape) > 3 :
        ax.imshow(volumen[:, volumen.shape[1] // y, :, b].T, interpolation=inter, cmap=cmap,origin="lower",vmin=vmin, vmax=vmax)
    else:
        ax.imshow(volumen[:, volumen.shape[1] // y].T, interpolation=inter, cmap=cmap, origin="lower",vmin=vmin, vmax=vmax)
    plt.axis('off')

    if title is not None:
        plt.title(title)

    rect_t = rect[1], rect[0], rect[3], rect[2]
    rectangle = patches.Rectangle((rect_t[0], rect_t[1]), rect_t[2], rect_t[3], linewidth=linewidth, edgecolor=edgecolor, facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rectangle)
    plt.show()

    print volumen.shape, 'y=', volumen.shape[1] // y

    return plt, fig, ax


def __plotOneV2(sp1,sp2, sp3, vol, y , b, inter, cmap, render=None, origin="upper", vmin=0, vmax=500):
    if render is None:
        render = plt
    if len(vol.shape) > 3:
        render.subplot(sp1,sp2, sp3).set_axis_off()
        return render.imshow(vol[:, vol.shape[1] // y, :, b], interpolation=inter, cmap=cmap, origin=origin, vmin=vim, vmax=vmax)
    else:
        render.subplot(sp1,sp2,sp3).set_axis_off()
        return render.imshow(vol[:, vol.shape[1] // y, :], interpolation=inter, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)


def _iswrc(vol1, vol2, rect1, rect2, y=2, b=0, inter='none', cmap='gray', title=None,linewidth=2, edgecolor='r',vmin=0, vmax=500):
    """
        idem iswrc con dos volumnes y dos rectangulos
    """
    fig, (ax1, ax2) = plt.subplots(2)

    __plotOneV2(1, 3, 1, vol1.T, y, b, inter, cmap, render=plt, origin="lower", vmin=vmin, vmax=vmax)
    __plotOneV2(1, 3, 2, vol2.T, y, b, inter, cmap, render=plt, origin="lower", vmin=vmin, vmax=vmax)

    plt.axis('off')

    if title is not None:
        plt.title(title)

    rect_t1 = rect1[1], rect1[0], rect1[3], rect1[2]
    rectangle1 = patches.Rectangle((rect_t1[0], rect_t1[1]), rect_t1[2], rect_t1[3], linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rectangle1)
    #plt.subplot(1,3,1).add_patch(rectangle1)

    rect_t2 = rect2[1], rect2[0], rect2[3], rect2[2]
    rectangle2 = patches.Rectangle((rect_t2[0], rect_t2[1]), rect_t2[2], rect_t2[3], linewidth=linewidth,
                                   edgecolor=edgecolor, facecolor='none')
    # Add the patch to the Axes
    ax2.add_patch(rectangle2)
    #plt.subplot(1, 3, 2).add_patch(rectangle2)

    plt.show()
    return plt, fig, (ax1, ax2)


def _is3d(volumen, y=2, b=0, inter='none', cmap='gray', vmin=0, vmax=500):
    plt.imshow(np.rot90(volumen[:, volumen.shape[1] // y, :]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    #plt.colorbar()
    return plt


def _isc(vol1, vol2, y=-1, yp=-1, b=0, inter='none', cmap='gray',titles=None, vmin=0, vmax=500):

    if y <= 0 and yp <= 0 :
        y1 = vol1.shape[1] // 2
        y2 = vol2.shape[1] // 2
    elif y >= 0:
        y1 = y2 = y
    else:
        y1 = vol1.shape[1] // yp
        y2 = vol2.shape[1] // yp

    # plt.figure('Showing the datasets')
    plt.subplot(1, 2, 1).set_axis_off()
    plt.imshow(np.rot90(vol1[:, y1, :, b]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 2).set_axis_off()
    plt.imshow(np.rot90(vol2[:, y2, :, b]), interpolation=inter, cmap=cmap, vmin=vmin, vmax=vmax)
    #plt.colorbar()
    if titles is not None:
        plt.subplot(1, 2, 1).set_title(titles[0])
        plt.subplot(1, 2, 2).set_title(titles[1])
    return plt

def _isc3(vol1, vol2, vol3, y=2, b=0, inter='none', cmap='gray', titles=None,vmin=None, vmax=None):
    # plt.figure('Showing the datasets')
    """
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(np.rot90(vol1[:, vol1.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(np.rot90(vol2[:, vol2.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(np.rot90(vol3[:, vol3.shape[1] // y, :, b]), interpolation=inter, cmap=cmap)
    """
    vmin1, vmin2, vmin3 = get_proper_value(vmin, 3)
    vmax1, vmax2, vmax3 =  get_proper_value(vmax, 3)

    im1 =__plotOne(1, 3, 1, vol1,y,b,inter,cmap,vmin=vmin1, vmax=vmax1)
    im2 =__plotOne(1, 3, 2, vol2,y,b,inter,cmap,vmin=vmin2, vmax=vmax2)
    im3= __plotOne(1, 3, 3, vol3,y,b,inter,'Blues',vmin=vmin3, vmax=vmax3)

    if titles is not None :
        plt.subplot(1, 3, 1).set_title(titles[0])
        plt.subplot(1, 3, 2).set_title(titles[1])
        plt.subplot(1, 3, 3).set_title(titles[2])
    plt.show()
    #plt.colorbar()
    return plt, im1,im2,im3

def get_proper_value(vlimit, amount_needed):
    if isinstance(vlimit, list):
        res = list((1,) * amount_needed)
        k = len(vlimit)-1
        for i in range(amount_needed-1, -1, -1):
            res[i] = vlimit[k]
            if k > 0 :
                k=k-1
        res = tuple(res)
    else:
        res = (vlimit, )*amount_needed
    return res

def __plotOne(sp1,sp2,sp3, vol, y , b, inter, cmap,vmin=0, vmax=500):
    if len(vol.shape) > 3:
        plt.subplot(sp1,sp2,sp3).set_axis_off()
        return plt.imshow(np.rot90(vol[:, vol.shape[1] // y, :, b]), interpolation=inter, cmap=cmap,vmin=vmin, vmax=vmax)
    else:
        plt.subplot(sp1,sp2,sp3).set_axis_off()
        return plt.imshow(np.rot90(vol[:, vol.shape[1] // y, :]), interpolation=inter, cmap=cmap,vmin=vmin, vmax=vmax)


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



def buildDownsamplig(shape, factor):
    """
    Build the downsamplig matrix for reduce (100/factor)% a 3d image
    use:
     D = buildDownsampligBy2(A.shape , factor)
     small = D.dot(A.reshape(-1)).reshape(A.shape/factor**3)
    :param nx, ny, nz:
    :return: sparce matrix (nx*ny*nz/factor**3, nx*ny*nz)
    """
    from scipy.sparse import lil_matrix
    nx, ny, nz = shape
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
