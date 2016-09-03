import numpy as np
import utils.img_utils as iu
import load.hcp_img_loader as hcp

A = np.random.random((126, 126, 126))*500

plt = iu._ig(A, './test.pdf', title='Cluster demo')
#plt.savefig('./test.pdf')

#x0,xf, y0,yf, z0,zf = (22, 28, 15, 21, 23, 29)
#rect_hr = 2*z0, 2*x0, 2*xf-2*x0, 2*zf-2*z0
#res=iu._iswr(A[:,2*y0:2*yf,:],rect_hr,title='Orig '+ str(A.shape))
#res[1].savefig('./test.pdf')

subjects = [100307, 100408, 180129, 180432, 180836, 180937]
img, gtab = hcp.load_subject_medium(0, subjects, base_folder='..')


plt = iu._ig(img.get_data(), './test.pdf', title='Cluster demo')
