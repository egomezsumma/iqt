import numpy as np
import nibabel as nib

from dipy.core.gradients import gradient_table
import sys

IN_NEF = '/home/lgomez/' in sys.prefix

# Carpeta
BVALS_FILE_NEF = '/data/athena/share/HCP/%s/T1w/Diffusion/bvals'
BVECS_FILE_NEF = '/data/athena/share/HCP/%s/T1w/Diffusion/bvecs'
# Nifty file
NIFTY_FILE_NEF ='/data/athena/share/HCP/%s/T1w/Diffusion/data.nii.gz'


def get_bvals(subject_str, folder=None):
    if IN_NEF :
        return np.loadtxt(BVALS_FILE_NEF%(subject_str))
    else:
        return np.loadtxt(folder + 'bvals_' + subject_str)

def get_bvecs(subject_str, folder=None):
    if IN_NEF :
        return np.loadtxt(BVECS_FILE_NEF%(subject_str))
    else:
        return np.loadtxt(folder + 'bvecs_' + subject_str)

def get_shape(subject):
    nx, ny, nz, nb = 12, 12, 12, 37
    if IN_NEF :
        src_name = NIFTY_FILE_NEF % (subject)
        img = nib.load(src_name)

        nx, ny, nz, nb = img.shape
    return nx, ny, nz, nb


def get_img(subject, file_name, bsize=-1, size=12, i=8, j=7, k=8):
    if IN_NEF :
        src_name = NIFTY_FILE_NEF % (subject)
        img = nib.load(src_name)

        nx, ny, nz, nb = img.shape

        x0, y0, z0 = int(size*i), int(size*j), int(size*k)
        xf, yf, zf = x0+size, y0+size, z0+size
    
        bsize = nb if bsize < 0 else bsize

        # Bval especifico
        # data = np.asarray(img.dataobj[x0:xf, y0:yf, z0:zf, bvals])
        # para todos y todas
        data = np.asarray(img.dataobj[x0:xf, y0:yf, z0:zf, 0:bsize])

        #print 'Final patch size:',  data.shape

        nifti1img = nib.Nifti1Image(data, img.affine)
        nib.load()
        del(img)
        return nifti1img
    else:
        return nib.load(file_name)

def load_subject_medium(index, numbers, bval=None, bvalpos=None, base_folder='.'):
    subject = str(numbers[index])

    folder = base_folder + '/HCP/' + subject + '/'
    print 'Apunto de cargar bvals'
    bvals = get_bvals(subject, folder)
    print 'Apunto de cargar bvecs'
    bvecs = get_bvecs(subject, folder)

    if bvalpos is not None:
        file_name = folder + 'data_medium40g_12x12x12x40_' + subject + '_b' + str(bval) + '.nii.gz'
        img = get_img(subject, file_name)
        bsize = min(len(bvalpos), img.shape[3])
        gtab = gradient_table(bvals=bvals[bvalpos[:bsize]], bvecs=bvecs[:, bvalpos[:bsize]])
    else:
        file_name = folder + 'data_medium40g_12x12x12x40_' + subject + '.nii.gz'
        img = get_img(subject, file_name)
        bsize = img.shape[3]
        gtab = gradient_table(bvals=bvals[:bsize], bvecs=bvecs[:, :bsize])
    return img, gtab


def load_subject_small(index, numbers, bval=None, bvalpos=None,base_folder='.'):
    subject = str(numbers[index])
    folder = base_folder +'/HCP/' + subject + '/'

    bvals = get_bvals(subject, folder)
    bvecs = get_bvecs(subject, folder)

    #print '#'+subject+':', bvals[:6]
    if bvalpos is not None:
        file_name = folder + 'data_small_12x12x12x6_' + subject + '_b' + str(bval) + '.nii.gz'
        img = get_img(subject, file_name)
        bsize = min(len(bvalpos), img.shape[3])
        gtab = gradient_table(bvals=bvals[bvalpos[:bsize]], bvecs=bvecs[:, bvalpos[:bsize]])
    else:
        file_name = folder + 'data_small_12x12x12x6_' + subject + '.nii.gz'
        img = get_img(subject, file_name)
        bsize = img.shape[3]
        gtab = gradient_table(bvals=bvals[:bsize], bvecs=bvecs[:, :bsize])
    return img, gtab


def load_subject_small_noS0(index, numbers, bval=None, bvalpos=None, base_folder='.'):
    subject = str(numbers[index])
    folder = base_folder +'/HCP/' + subject + '/'
    
    bvals = get_bvals(subject, folder)
    bvecs = get_bvecs(subject, folder)

    #print '#'+subject+':', bvals[:6]
    if bvalpos is not None:
        file_name = folder + 'data_small_12x12x12x6_' + subject + '_b' + str(bval) + '.nii.gz'
        img = get_img(subject, file_name)
        bsize = min(len(bvalpos), img.shape[3])
        bs  = bvals[bvalpos[:bsize]]
        bvs = bvecs[:, bvalpos[:bsize]]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:,idxs])
    else:
        img = folder + 'data_small_12x12x12x6_' + subject + '.nii.gz'
        img = get_img(subject, file_name)
        bsize = img.shape[3]
        bs = bvals[:bsize]
        bvs = bvecs[:, :bsize]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:,idxs])
    return img, gtab, idxs

def __index_not_equals_to(arr, values):
    return [ i for i in xrange(len(arr)) if arr[i] not in values]


def load_subject_medium_noS0(subject_number, bval=None, bvalpos=None, bsize=-1, base_folder='.'):
    subject = str(subject_number)
    folder = base_folder +'/HCP/' + subject + '/'

    print 'Apunto de cargar bvals', subject
    bvals = get_bvals(subject, folder)
    print 'Apunto de cargar bvecs', subject
    bvecs = get_bvecs(subject, folder)

    if bvalpos is not None:
        file_name = folder + 'data_medium40g_12x12x12x40_' + subject + '_b' + str(bval) + '.nii.gz'
        img = get_img(subject, file_name, bsize=bsize)
        bsize = min(len(bvalpos), img.shape[3])
        bs = bvals[bvalpos[:bsize]]
        bvs = bvecs[:, bvalpos[:bsize]]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:,idxs])
    else:
        file_name = folder + 'data_medium40g_12x12x12x40_' + subject + '.nii.gz'
        print 'Apunto de cargar patch ', subject
        img = get_img(subject, file_name, bsize=bsize)
        bsize = img.shape[3]
        bs = bvals[:bsize]
        bvs = bvecs[:, :bsize]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:, idxs])
    return img, gtab, idxs


