import mymapl.mapmri as mp
import numpy as np

def getC(data, gtab, radial_order=4):
    map_model = mp.MapmriModel(gtab,
                               radial_order=radial_order,
                               laplacian_regularization=True,
                               laplacian_weighting=0.2,
                               anisotropic_scaling=False,
                               dti_scale_estimation=False)
    map_model_fit = map_model.fit(data)
    # (Nx, Ny, Nz, Nc)
    C = map_model_fit.mapmri_coeff
    return C


def getMAndU(radial_order, mu, gtab, tau):
    r'''Recovers the reconstructed signal for any qvalue array or
        gradient table.
        '''
    qvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
    q = qvals[:, None] * gtab.bvecs
    M = mp.mapmri_isotropic_phi_matrix(radial_order, mu[0], q)

    # Lapalcian matrix
    laplacian_matrix = mp.mapmri_isotropic_laplacian_reg_matrix(radial_order, 1.)
    laplacian_matrix = laplacian_matrix * mu[0]

    return M, laplacian_matrix


def getM(radial_order, mu, gtab, tau):
    r'''Recovers the reconstructed signal for any qvalue array or
        gradient table.
        '''
    qvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
    q = qvals[:, None] * gtab.bvecs
    M = mp.mapmri_isotropic_phi_matrix(radial_order, mu[0], q)
    return M


def get_mapl_params(gtab, radial_order=4):
    # Fiteo el model
    map_model = mp.MapmriModel(gtab,
                               radial_order=radial_order,
                               laplacian_regularization=True,
                               laplacian_weighting=0.2,
                               anisotropic_scaling=False,
                               dti_scale_estimation=False)
    # Fiteo la data
    # map_model_fit = map_model.fit(i_hr)
    tau = map_model.tau
    mu = map_model.mu
    #print 'mu.shape', mu.shape
    M = getM(radial_order, mu, gtab, tau)
    return M, tau, mu


def get_mapl_params2(gtab, radial_order=4):
    # Fiteo el model
    map_model = mp.MapmriModel(gtab,
                               radial_order=radial_order,
                               laplacian_regularization=True,
                               laplacian_weighting=0.2,
                               anisotropic_scaling=False,
                               dti_scale_estimation=False)
    # Fiteo la data
    # map_model_fit = map_model.fit(i_hr)
    tau = map_model.tau
    mu = map_model.mu
    #print 'mu.shape', mu.shape, mu
    M, U = getMAndU(radial_order, mu, gtab, tau)
    return M, tau, mu, U
