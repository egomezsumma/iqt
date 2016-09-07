#!/home/lgomez/anaconda2/bin/python
#OAR -l {mem>=200000}/nodes=4/core=12,walltime=96

#import hcp_img_loader as load
import sys
import numpy as np
import nibabel as nib



IS_NEF = '/home/lgomez/' in sys.prefix


if IS_NEF :
    subjects = list(np.loadtxt('/home/lgomez/demo/100sujetos.txt', dtype='int'))
else:
    subjects = [100307, 100408, 180129, 180432, 180836, 180937]


def get_img(subject):
    NIFTY_FILE_NEF = '/data/athena/share/HCP/%s/T1w/Diffusion/data.nii.gz'
    src_name = NIFTY_FILE_NEF % (subject)
    img = nib.load(src_name)
    return img

def hashshape(a_shape):
    return '%d_%d_%d_%d'%a_shape

shapes={}
no_andan=[]
for s in subjects:
    try:
        img = get_img(s)
        hash = hashshape(img.shape)
        if hash not in shapes :
            shapes[hash] = []
        shapes[hash].append(s)

    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        no_andan.append(s)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        no_andan.append(s)

print 'no anda', no_andan
print 'shapes:', len(shapes.keys()), 'types of sizes'
for k in shapes.keys():
    print k,'(',len(shapes[k]), '):', shapes[k]

