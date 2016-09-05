from load.hcp_img_loader import load_subject_medium_noS0
import sys
import numpy as np

IS_NEF = '/home/lgomez/' in sys.prefix


if IS_NEF :
    subjects = list(np.loadtxt('/home/lgomez/demo/50sujetos.txt', dtype='int'))
else:
    subjects = [100307, 100408, 180129, 180432, 180836, 180937]


no_andan=[]
andan=[]
for s in subjects:
    try:
        img, gtab, idxs = load_subject_medium_noS0(s, bsize=55)
        if 1000 not in gtab.bvals:
	        print s, 'no tiene bval 1000', set(gtab.bvals)
	        no_andan.append(s)
        if 2000 not in gtab.bvals:
            print s, 'no tiene bval 2000',set(gtab.bvals)
            no_andan.append(s)
        if 3000 not in gtab.bvals:
            print s, 'no tiene bval 3000', set(gtab.bvals)
            no_andan.append(s)
        if s not in no_andan:
            andan.append(s)
        else:
            print 'sujeto', s, 'cumple requisitos'
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        no_andan.append(s)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        no_andan.append(s)

print 'no anda', no_andan
print 'andan', andan