from cvxpy.expressions.expression import Expression
import cvxpy as cvx

if cvx.__version__ == '0.3.1':
    from cvxpy.atoms.elementwise.norm2_elemwise import norm2_elemwise



def tv3d_CubosYSlices(value, *args):
    value = Expression.cast_to_const(value)
    rows, cols = value.size
    if value.is_scalar():
        raise ValueError("tv cannot take a scalar argument.")
    # L1 norm for vectors.
    elif value.is_vector():
        return ValueError("You may call tv.")
    # L2 norm for matrices.
    else:
        args = map(Expression.cast_to_const, args)
        slices = [value] + list(args)

        Nz = len(slices)
        Nx, Ny = slices[0].size
        if Nz < 2 or Nx < 2 or Ny < 2:
            return ValueError("You may call tv.")

        norm2_slices = []
        cub_k, cub_x, cub_y, cub_z = [], [], [], []

        for z in range(0, Nz - 1):
            mat = slices[z]
            cub_k.append(cvx.vec(mat[0:Nx - 1, 0:Ny - 1]))
            cub_x.append(cvx.vec(mat[1:Nx, 0:Ny - 1]))
            cub_y.append(cvx.vec(mat[0:Nx - 1, 1:Ny]))

        for z in range(1, Nz):
            mat = slices[z]
            cub_z.append(cvx.vec(mat[0:Nx - 1, 0:Ny - 1]))

        K = cvx.vstack(*cub_k)
        X = cvx.vstack(*cub_x)
        Y = cvx.vstack(*cub_y)
        Z = cvx.vstack(*cub_z)

        diffs = []
        print 'using cvxpy version:', cvx.__version__
        if cvx.__version__ == '0.3.1':
            diffs += [
                X - K,
                Y - K,
                Z - K,
            ]
            # Norm2ElementWis (deprecada) en 0.4.8
            norm2 = norm2_elemwise(*diffs)
        else:
            diffs += [
                (X - K)**2,
                (Y - K)**2,
                (Z - K)**2,
            ]
            norm2 = cvx.sqrt(sum(diffs))

        return cvx.sum_entries(norm2)


def call_tv3d_CubosYSlices(Yhr, Nx, Ny, Nz, bval):
    vhr = Nx * Ny * Nz
    vhrb = vhr * bval
    Nxy = Nx * Ny
    b_offset = vhr
    tvs_by_b = []
    for b in xrange(bval):
        b_offset = b * vhr
        slices = []
        for z in range(0, Nz):
            XY = Yhr[b_offset + z * Nxy: b_offset + (z + 1) * Nxy]  # el slice XY para este z y b
            mat = cvx.reshape(XY, Nx, Ny)
            slices.append(mat)

        tvs_by_b.append(tv3d_CubosYSlices(*slices))
        # tvs_by_b.append(cvx.tv3d(*slices))

    print '#slices', len(slices)
    return sum(tvs_by_b)


def tv3d(Yhr, Nx, Ny, Nz, Nb):
    return call_tv3d_CubosYSlices(Yhr, Nx, Ny, Nz, Nb)