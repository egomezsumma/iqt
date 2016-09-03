import numpy as np
import nibabel as nib

from dipy.core.gradients import gradient_table


def load_subject_medium(index, numbers, bval=None, bvalpos=None, base_folder='.'):
    subject = str(numbers[index])

    folder = base_folder + '/HCP/' + subject + '/'

    bvals = np.loadtxt(folder + 'bvals_' + subject)
    bvecs = np.loadtxt(folder + 'bvecs_' + subject)

    if bvalpos is not None:
        img = nib.load(folder + 'data_medium40g_12x12x12x40_' + subject + '_b' + str(bval) + '.nii.gz')
        bsize = min(len(bvalpos), img.shape[3])
        gtab = gradient_table(bvals=bvals[bvalpos[:bsize]], bvecs=bvecs[:, bvalpos[:bsize]])
    else:
        img = nib.load(folder + 'data_medium40g_12x12x12x40_' + subject + '.nii.gz')
        bsize = img.shape[3]
        gtab = gradient_table(bvals=bvals[:bsize], bvecs=bvecs[:, :bsize])
    return img, gtab


def load_subject_small(index, numbers, bval=None, bvalpos=None,base_folder='.'):
    subject = str(numbers[index])
    folder = base_folder +'/HCP/' + subject + '/'
    bvals = np.loadtxt(folder + 'bvals_' + subject)
    bvecs = np.loadtxt(folder + 'bvecs_' + subject)

    print '#'+subject+':', bvals[:6]
    if bvalpos is not None:
        img = nib.load(folder + 'data_small_12x12x12x6_' + subject + '_b' + str(bval) + '.nii.gz')
        bsize = min(len(bvalpos), img.shape[3])
        gtab = gradient_table(bvals=bvals[bvalpos[:bsize]], bvecs=bvecs[:, bvalpos[:bsize]])
    else:
        img = nib.load(folder + 'data_small_12x12x12x6_' + subject + '.nii.gz')
        bsize = img.shape[3]
        gtab = gradient_table(bvals=bvals[:bsize], bvecs=bvecs[:, :bsize])
    return img, gtab


def load_subject_small_noS0(index, numbers, bval=None, bvalpos=None, base_folder='.'):
    subject = str(numbers[index])
    folder = base_folder +'/HCP/' + subject + '/'
    bvals = np.loadtxt(folder + 'bvals_' + subject)
    bvecs = np.loadtxt(folder + 'bvecs_' + subject)

    #print '#'+subject+':', bvals[:6]
    if bvalpos is not None:
        img = nib.load(folder + 'data_small_12x12x12x6_' + subject + '_b' + str(bval) + '.nii.gz')
        bsize = min(len(bvalpos), img.shape[3])
        bs  = bvals[bvalpos[:bsize]]
        bvs = bvecs[:, bvalpos[:bsize]]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:,idxs])
    else:
        img = nib.load(folder + 'data_small_12x12x12x6_' + subject + '.nii.gz')
        bsize = img.shape[3]
        bs = bvals[:bsize]
        bvs = bvecs[:, :bsize]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:,idxs])
    return img, gtab, idxs

def __index_not_equals_to(arr, values):
    return [ i for i in xrange(len(arr)) if arr[i] not in values]


def load_subject_medium_noS0(index, numbers, bval=None, bvalpos=None, base_folder='.'):
    subject = str(numbers[index])
    folder = base_folder +'/HCP/' + subject + '/'
    bvals = np.loadtxt(folder + 'bvals_' + subject)
    bvecs = np.loadtxt(folder + 'bvecs_' + subject)

    if bvalpos is not None:
        img = nib.load(folder + 'data_medium40g_12x12x12x40_' + subject + '_b' + str(bval) + '.nii.gz')
        bsize = min(len(bvalpos), img.shape[3])
        bs = bvals[bvalpos[:bsize]]
        bvs = bvecs[:, bvalpos[:bsize]]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:,idxs])
    else:
        img = nib.load(folder + 'data_medium40g_12x12x12x40_' + subject + '.nii.gz')
        bsize = img.shape[3]
        bs = bvals[:bsize]
        bvs = bvecs[:, :bsize]
        idxs = __index_not_equals_to(bs, [0, 5])
        gtab = gradient_table(bvals=bs[idxs], bvecs=bvs[:, idxs])
    return img, gtab, idxs
