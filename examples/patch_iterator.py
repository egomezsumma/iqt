


import numpy as np
n=2
m=2


patchN1 = 2*n+1
patchN2 = m

shapeLrImg = (patchN1*3, patchN1*3, patchN1)
shapeHrImg = (patchN1*3*2, patchN1*3*2, patchN1*2)

lrimg = np.arange(patchN1*3*patchN1*3*patchN1).reshape(shapeLrImg);
hrimg = np.zeros((patchN1*3*2,patchN1*3*2,patchN1*2))#.reshape(shapeHrImg);


sum = 0; sum_acum =0
for i in range(0,shapeLrImg[0]-patchN1+1):
    for j in range(0,shapeLrImg[1]-patchN1+1):
        for k in range(0,shapeLrImg[2]-patchN1+1) :
            #print i,j,k, '-->', lrimg[i][j][k];

            sum=sum+1
            x = (i+2)*patchN2
            y = (j+2) * patchN2
            z = (k+2) * patchN2

            unos = np.ones((patchN2, patchN2, patchN2))
            sum_acum = sum_acum+ unos.sum()
            hrimg[x:x+patchN2, y:y+patchN2, z:z+patchN2] = unos

            print i, ':', i + patchN1,  j, ':', j + patchN1, k, ':', k + patchN1, ' -->', \
                x, ':', x + patchN2, y, ':', y + patchN2, z, ':', z + patchN2;
print 'iteratiosn=s', sum, sum*(patchN2**3), sum_acum

print lrimg.shape, '-->', hrimg.shape
print hrimg.sum();



from utils.dmri_patch_operations.LrHrPatchIterator import LrHrPatchIterator;

it = LrHrPatchIterator(shapeLrImg,n,m)

ranges = [x for x in it]
print ranges




