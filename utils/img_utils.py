

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