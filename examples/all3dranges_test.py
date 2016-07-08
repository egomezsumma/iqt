from utils.RandomRanges import All3DRangePosibleNotOverlapping
import numpy as np

shapelr = (82, 106, 76)
shapelr = (82/2, 106/2, 76/2)
shape = shapelr
a = np.random.random(shape)
it = All3DRangePosibleNotOverlapping(shape, 5, 5, 5)


sum=0
for t in it:
    #x0, xf, y0, yf, z0, zf = t
    #print a[x0:xf, y0:yf, z0:zf]
    sum =sum+1


print sum
