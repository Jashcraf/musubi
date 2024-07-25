import numpy as np
import scipy.linalg as la

C = 299_792_458  # m / s
e0 = 8.854187817620389e-12
m0 = 4*np.pi*1.0e-7
z0 = 3.767303134617706e2 # sqrt(mu0 / e0)

def wavelength_to_angular_frequency(wvl):
    """Convert wavelength to angular frequency

    Parameters
    ----------
    wvl : float
        wavelength of light

    Returns
    -------
    float
        angular frequency of light (freq * 2 * pi)
    """
    return C / wvl * 2 * np.pi


def _empty_tensor(shape):
    """Returns an empty array to populate with a dielectric tensor.

    Parameters
    ----------
    shape : list
        shape to prepend to the dielectric tensor array. shape = [32,32] returns
        an array of shape [32,32,3,3] where the matrix is assumed to be in the
        last indices. Defaults to None, which returns a 3x3 array.

    Returns
    -------
    numpy.ndarray
        The zero array of specified shape

    Notes
    -----
    The structure of this function was taken from prysm.x.polarization,
    which was written by Jaren Ashcraft
    """

    if shape is None:

        shape = (3, 3)

    else:

        shape = (*shape, 3, 3)

    return np.zeros(shape)


def _empty_berreman(shape):
    """Returns an empty array to populate with a berreman field matrix.

    Parameters
    ----------
    shape : list
        shape to prepend to the dielectric tensor array. shape = [32,32] returns
        an array of shape [32,32,4,4] where the matrix is assumed to be in the
        last indices. Defaults to None, which returns a 4x4 array.

    Returns
    -------
    numpy.ndarray
        The zero array of specified shape

    Notes
    -----
    The structure of this function was taken from prysm.x.polarization,
    which was written by Jaren Ashcraft
    """

    if shape is None:

        shape = (4, 4)

    else:

        shape = (*shape, 4, 4)

    return np.zeros(shape)


def _empty_maxwell(shape):
    """Returns an empty array to populate the matrix form of Maxwell's Equations.

    Parameters
    ----------
    shape : list
        shape to prepend to the maxwell tensor array. shape = [32,32] returns
        an array of shape [32,32,6,6] where the matrix is assumed to be in the
        last indices. Defaults to None, which returns a 6x6 array.

    Returns
    -------
    numpy.ndarray
        The zero array of specified shape

    Notes
    -----
    The structure of this function was taken from prysm.x.polarization,
    which was written by Jaren Ashcraft
    """

    if shape is None:

        shape = (6, 6)

    else:

        shape = (*shape, 6, 6)

    return np.zeros(shape)


def dielectric_tensor(e1, e2, e3, shape=None):
    """Construct a dielectric tensor from the dielectric constants, also works for
    the permeability constants for the magnetic field

    Parameters
    ----------
    e1 : float
        dielectric constant associated with the optic axis
    e2 : float
        dielectric constant associated with the second principal axis
    e3 : float
        dielectric constant associated with the third principal axis
    shape : list, optional
        shape to prepend to the delectric tensor array, see `_empty_tensor`.
        By default None

    Returns
    -------
    numpy.ndarray
        array of dielectric tensors
    """

    tensor = _empty_tensor(shape)
    tensor[..., 0, 0] = e1
    tensor[..., 1, 1] = e2
    tensor[..., 2, 2] = e3

    return tensor


def rotation_x(theta_x, shape=None):

    R = _empty_tensor(shape)
    cost = np.cos(theta_x)
    sint = np.sin(theta_x)

    R[..., 0, 0] = 1
    R[..., 1, 1] = cost
    R[..., 2, 2] = cost

    R[..., 1, 2] = -sint
    R[..., 2, 1] = sint

    return R


def rotation_y(theta_y, shape=None):

    R = _empty_tensor(shape)
    cost = np.cos(theta_y)
    sint = np.sin(theta_y)

    R[..., 0, 0] = cost
    R[..., 1, 1] = 1
    R[..., 2, 2] = cost

    R[..., 0, 2] = -sint
    R[..., 2, 0] = sint

    return R


def rotation_z(theta_z, shape=None):

    R = _empty_tensor(shape)
    cost = np.cos(theta_z)
    sint = np.sin(theta_z)

    R[..., 0, 0] = cost
    R[..., 1, 1] = cost
    R[..., 2, 2] = 1

    R[..., 0, 1] = -sint
    R[..., 1, 0] = sint

    return R


def rotate_xyz(M, eta, psi, xi, shape=None):

    # NOTE: This is how the 3D rotation is written in the matlab pckg
    # TODO: write test to ensure that this works

    Rpx_eta = rotation_x(eta, shape=shape)
    Rpz_psi = rotation_z(psi, shape=shape)
    Rpx_xi = rotation_x(xi, shape=shape)

    Rmx_eta = rotation_x(-eta, shape=shape)
    Rmz_psi = rotation_z(-psi, shape=shape)
    Rmx_xi = rotation_x(-xi, shape=shape)

    return Rpx_xi @ Rpz_psi @ Rpx_eta @ M @ Rmx_eta @ Rmz_psi @ Rmx_xi


def rotation_3d(tensor, theta_x, theta_z, theta_x2=0, shape=None):
    """Rotate a dielectric tensor in 3D

    Parameters
    ----------
    tensor : numpy.ndarray
        the diagonal dielectric tensor to rotate
    theta_x : float
        rotation angle around x axis, first rotation
    theta_z : float
        rotation angle around z axis
    theta_x2 : float
        rotation angle around x axis, second rotation, by default 0
    shape : list, optional
        shape to prepend to the delectric tensor array, see `_empty_tensor`.
        By default None

    Returns
    -------
    numpy.ndarray
        the rotated tensor about the specified angles
    """

    R_in = rotation_matrix_3d(theta_x, theta_x2, theta_z, shape=shape)
    R_out = rotation_matrix_3d(-theta_x, -theta_x2, -theta_z, shape=shape)

    return R_in @ tensor @ R_out


def S_from_M(M, wavelength, ambient_index, aoi):

    # this part is from Azzam "Ellipsometry and Polarized Light"
    frequency = wavelength_to_angular_frequency(wavelength)
    sinaoi = np.sin(aoi)
    xi = (frequency / C ) * ambient_index * sinaoi
    eta = xi / frequency

    M11 = M[..., 0, 0]
    M12 = M[..., 0, 1]
    M13 = M[..., 0, 2]
    M14 = M[..., 0, 3]
    M15 = M[..., 0, 4]
    M16 = M[..., 0, 5]

    M21 = M[..., 1, 0]
    M22 = M[..., 1, 1]
    M23 = M[..., 1, 2]
    M24 = M[..., 1, 3]
    M25 = M[..., 1, 4]
    M26 = M[..., 1, 5]

    M31 = M[..., 2, 0]
    M32 = M[..., 2, 1]
    M33 = M[..., 2, 2]
    M34 = M[..., 2, 3]
    M35 = M[..., 2, 4]
    M36 = M[..., 2, 5]

    M41 = M[..., 3, 0]
    M42 = M[..., 3, 1]
    M43 = M[..., 3, 2]
    M44 = M[..., 3, 3]
    M45 = M[..., 3, 4]
    M46 = M[..., 3, 5]

    M51 = M[..., 4, 0]
    M52 = M[..., 4, 1]
    M53 = M[..., 4, 2]
    M54 = M[..., 4, 3]
    M55 = M[..., 4, 4]
    M56 = M[..., 4, 5]

    M61 = M[..., 5, 0]
    M62 = M[..., 5, 1]
    M63 = M[..., 5, 2]
    M64 = M[..., 5, 3]
    M65 = M[..., 5, 4]
    M66 = M[..., 5, 5]

    # Eq 21 Berreman 1972
    d = M33 * M66 - M36 * M63

    a31 = (M61 * M36 - M31 * M66) / d
    a32 = ((M62 - eta) * M36 - M32 * M66) / d
    a34 = (M64 * M36 - M34 * M66) / d
    a35 = (M65 * M36 - (M35 + eta) * M66) / d
    a61 = (M63 * M31 - M33 * M61) / d
    a62 = (M63 * M32 - M33 * (M62 - eta)) / d
    a64 = (M63 * M34 - M33 * M64) / d
    a65 = (M63 * (M35 + eta) - M33 * M65) / d

    # Eq 24 
    S11 = M11 + M13 * a31 + M16 * a61
    S12 = M12 + M13 * a32 + M16 * a62
    S13 = M14 + M13 * a34 + M16 * a64
    S14 = M15 + M13 * a35 + M16 * a65
    S21 = M21 + M23 * a31 + (M26 - cxiw) * a61
    S22 = M22 + M23 * a32 + (M26 - cxiw) * a61
    S23 = M24 + M23 * a34 + (M26 - cxiw) * a64
    S24 = M25 + M23 * a35 + (M26 - cxiw) * a65
    S31 = M41 + M43 * a31 + M46 * a61
    S32 = M42 + M43 * a32 + M46 * a62
    S33 = M44 + M43 * a34 + M46 * a64
    S34 = M45 + M43 * a35 + M46 * a65
    S41 = M51 + (M53 + cxiw) * a31 + M56 * a61
    S42 = M52 + (M53 + cxiw) * a32 + M56 * a62
    S43 = M54 + (M53 + cxiw) * a34 + M56 * a64
    S44 = M55 + (M53 + cxiw) * a35 + M56 * a65

    S = _empty_tensor(*M.shape[:-2])

    S[..., 0, 0] = S11
    S[..., 0, 1] = S12
    S[..., 0, 2] = S13
    S[..., 0, 3] = S14

    S[..., 1, 0] = S21
    S[..., 1, 1] = S22
    S[..., 1, 2] = S23
    S[..., 1, 3] = S24

    S[..., 2, 0] = S31
    S[..., 2, 1] = S32
    S[..., 2, 2] = S33
    S[..., 2, 3] = S34

    S[..., 3, 0] = S41
    S[..., 3, 1] = S42
    S[..., 3, 2] = S43
    S[..., 3, 3] = S44

    return S


def Delta_from_M(M, wavelength, ambient_index, aoi):

    # this part is from Azzam "Ellipsometry and Polarized Light"
    frequency = wavelength_to_angular_frequency(wavelength)
    sinaoi = np.sin(aoi)
    xi = (frequency / C ) * ambient_index * sinaoi
    eta = xi / frequency

    M11 = M[..., 0, 0]
    M12 = M[..., 0, 1]
    M13 = M[..., 0, 2]
    M14 = M[..., 0, 3]
    M15 = M[..., 0, 4]
    M16 = M[..., 0, 5]

    M21 = M[..., 1, 0]
    M22 = M[..., 1, 1]
    M23 = M[..., 1, 2]
    M24 = M[..., 1, 3]
    M25 = M[..., 1, 4]
    M26 = M[..., 1, 5]

    M31 = M[..., 2, 0]
    M32 = M[..., 2, 1]
    M33 = M[..., 2, 2]
    M34 = M[..., 2, 3]
    M35 = M[..., 2, 4]
    M36 = M[..., 2, 5]

    M41 = M[..., 3, 0]
    M42 = M[..., 3, 1]
    M43 = M[..., 3, 2]
    M44 = M[..., 3, 3]
    M45 = M[..., 3, 4]
    M46 = M[..., 3, 5]

    M51 = M[..., 4, 0]
    M52 = M[..., 4, 1]
    M53 = M[..., 4, 2]
    M54 = M[..., 4, 3]
    M55 = M[..., 4, 4]
    M56 = M[..., 4, 5]

    M61 = M[..., 5, 0]
    M62 = M[..., 5, 1]
    M63 = M[..., 5, 2]
    M64 = M[..., 5, 3]
    M65 = M[..., 5, 4]
    M66 = M[..., 5, 5]

    # Eq 21 Berreman 1972
    d = M33 * M66 - M36 * M63

    a31 = (M61 * M36 - M31 * M66) / d
    a32 = ((M62 - eta) * M36 - M32 * M66) / d
    a34 = (M64 * M36 - M34 * M66) / d
    a35 = (M65 * M36 - (M35 + eta) * M66) / d
    a61 = (M63 * M31 - M33 * M61) / d
    a62 = (M63 * M32 - M33 * (M62 - eta)) / d
    a64 = (M63 * M34 - M33 * M64) / d
    a65 = (M63 * (M35 + eta) - M33 * M65) / d

    # Eq 24 in Berreman. Out of order to correspond to S
    D21 = M11 + M13 * a31 + M16 * a61
    D23 = M12 + M13 * a32 + M16 * a62
    D24 = -1 * (M14 + M13 * a34 + M16 * a64)
    D22 = M15 + M13 * a35 + M16 * a65
    D41 = M21 + M23 * a31 + (M26 - eta) * a61
    D43 = M22 + M23 * a32 + (M26 - eta) * a61
    D44 = -1 * (M24 + M23 * a34 + (M26 - eta) * a64)
    D42 = M25 + M23 * a35 + (M26 - eta) * a65
    D31 = -1 * (M41 + M43 * a31 + M46 * a61)
    D33 = -1 * (M42 + M43 * a32 + M46 * a62)
    D34 = M44 + M43 * a34 + M46 * a64
    D32 = -1 * (M45 + M43 * a35 + M46 * a65)
    D11 = M51 + (M53 + eta) * a31 + M56 * a61
    D13 = M52 + (M53 + eta) * a32 + M56 * a62
    D14 = -1 * (M54 + (M53 + eta) * a34 + M56 * a64)
    D12 = M55 + (M53 + eta) * a35 + M56 * a65

    D = _empty_berreman(M.shape[:-2])

    D[..., 0, 0] = D11
    D[..., 0, 1] = D12
    D[..., 0, 2] = D13
    D[..., 0, 3] = D14

    D[..., 1, 0] = D21
    D[..., 1, 1] = D22
    D[..., 1, 2] = D23
    D[..., 1, 3] = D24

    D[..., 2, 0] = D31
    D[..., 2, 1] = D32
    D[..., 2, 2] = D33
    D[..., 2, 3] = D34

    D[..., 3, 0] = D41
    D[..., 3, 1] = D42
    D[..., 3, 2] = D43
    D[..., 3, 3] = D44

    return D


def construct_M(wavelength, dielectric, permeability, shape=None, model=None, gamma=0):

    frequency = wavelength_to_angular_frequency(wavelength)
    M = _empty_maxwell(shape)

    epsilon = _empty_tensor(shape)
    mu = _empty_tensor(shape)

    # create diagonal dielectric constant matrices
    for i, (e, m) in enumerate(zip(dielectric, permeability)):
        epsilon[..., i, i] = e
        mu[..., i, i] = m

    M[..., :3, :3] = epsilon
    M[..., 3:, 3:] = mu

    # TODO: Support optical activity with nonzero diagonals
    # Drude model for isotropic materials, Berreman Eq 12
    if model == 'Drude':

        rho = -1j * gamma * (frequency / C)
        for i in range(3):
            M[..., i, 3+i] = rho

    # Born's model for faraday rotation in isotropic materials in the presence of
    # magnetic fields, Berreman Eq 13
    elif model == 'Faraday':

        M[..., 0, 1] = -1j * gamma
        M[..., 1, 0] = 1j * gamma

    return M


def construct_field_matrix(wavelength, dielectric, permeability, film_thickness, ambient_index=1, aoi=0, shape=None, model=None, gamma=0):

    M = construct_M(wavelength, dielectric, permeability, shape=shape, model=model, gamma=gamma)
    D = Delta_from_M(M, wavelength, ambient_index, aoi)

    # Solve the eigenvalue equation
    # NOTE: Defaults to right eigenvectors
    evals, right_evecs = np.linalg.eig(D)

    evalmat = np.zeros_like(right_evecs, dtype=np.complex128)
    for i in range(4):
        evalmat[..., i, i] = np.exp(1j * evals[i] * film_thickness)

    # TODO: Investigate if .inv can be .T. I suspect not because the eigenpolarizations of
    # anisotropic thin films are not necessarilly orthogonal
    field_matrix = right_evecs @ evalmat @ np.linalg.inv(right_evecs)

    return field_matrix


def reflection_from_field_matrix(F):

    M11 = F[:2, :2]
    M21 = F[2:, :2]
    R = -M21 @ np.linalg.inv(M11)

    return R


"""
Here begins the scripts that are copied over from the textbook's MATLAB files
"""

INDEX_TOL = 0.000001

def epsilon(param1):

    # in material format
    # param1 = [n1, n2, n3, eta, psi, xi]
    # ------------------
    # in layer format
    # param1 = [n1, n2, n3, eta, psi, xi]
    # ------------------

    e = np.array([[param1[0]**2, 0, 0],
                  [0, param1[1]**2, 0],
                  [0, 0, param1[2]**2]])
    
    y = rotate_xyz(e, param1[3], param1[4], param1[5])

    return y


def poynting(E, H=None):

    if H is not None:
        # This is just a cross product
        px = (E[..., 1, :].real * np.conj(H[..., 2, :])) - (E[..., 2, :] * np.conj(H[..., 1, :]))
        py = (E[..., 2, :].real * np.conj(H[..., 0, :])) - (E[..., 0, :] * np.conj(H[..., 2, :]))
        pz = (E[..., 0, :].real * np.conj(H[..., 1, :])) - (E[..., 1, :] * np.conj(H[..., 0, :]))

        p = np.array([px, py, pz]) / 2

    else:
        p = E[..., 0, :] * np.conj(E[..., 1, :]) - E[..., 2, :] * np.conj(E[..., 3, :])
        p = np.real(p) / 2

    return p



def field_matrix(arg1, beta=None):

    if beta == None:

        ny = arg1[0]
        nz = arg1[1]

        nyz0 = ny / z0
        nzz0 = nz / z0

        F = np.array([[1, 1, 0, 0],
                      [nyz0, -nyz0, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, -nzz0, nzz0]])
        
        alpha = [ny, -ny, nz, -nz]

        E = np.array([[0, 0, 0, 0],
                      [F[0]],
                      [F[2]]])
        
        H = np.array([[0, 0, 0, 0],
                      [F[3]],
                      [F[1]]])
        
    else:

        n1 = arg1[0]
        n2 = arg1[1]
        n3 = arg1[2]

        eta = arg1[3]
        psi = arg1[4]
        xi = arg1[5]

        # Treat as isotropic
        if (np.abs(n1-n2) < INDEX_TOL) and (np.abs(n1-n3) < INDEX_TOL):
            
            alph = np.sqrt(n1**2 - beta**2)
            gs = -alph / z0
            gp = n1**2 / alph / z0

            F = np.array([[1, 1, 0, 0],
                          [gp, -gp, 0, 0],
                          [0, 0, 1, 1],
                          [0, 0, gs, -gs]])
            
            alpha = [alph -alph, alph, -alph]
            exx = n1**2
            exy = 0
            exz = 0

        # Treat as anisotropic
        else:

            e = epsilon(arg1)

            # extract components of dielectric tensor
            exx = e[..., 0, 0]
            exy = e[..., 0, 1]
            exz = e[..., 0, 2]

            eyy = e[..., 1, 1]
            eyz = e[..., 1, 2]

            ezz = e[..., 2, 2]

            zeros = np.zeros_like(exx)

            # Build Mbeta
            M11 = -beta * exy / exx
            M12 = z0 * (1 - (beta**2 / exx))
            M13 = -beta * exz / exx
            M14 = zeros

            M21 = (eyy - (exy**2 / exx)) / z0
            M22 = -beta*exy/exx
            M23 = (eyz - (exy*exz / exx)) / z0
            M24 = zeros

            M31 = zeros
            M32 = zeros
            M33 = zeros
            M34 = -z0

            M41 = (-eyz + (exz * exy / exx)) / z0
            M42 = beta * exz / exx
            M43 = (beta**2 - ezz + (exz**2 / exx)) / z0
            M44 = zeros

            Mbeta = np.array([[M11, M12, M13, M14],
                              [M21, M22, M23, M24],
                              [M31, M32, M33, M34],
                              [M41, M42, M43, M44]])
            
            alpha, F = la.eig(Mbeta, right=True, left=False)
            print(alpha)
            print(F)

            # Normalize columns of field matrix
            f0 = F[0]
            f2 = F[2]

            normalize_columns = np.sqrt(f0 * f0.conj() + f2 * f2.conj())
            normalize_columns = normalize_columns * np.exp(1j * np.angle(f0 + f2))

            F = F / normalize_columns

            # TODO: Check similarity of numpy v.s. matlab sorting
            a = alpha
            a = np.sort(np.real(a))
            j = np.where(a)[0]
            a = np.array([a[j[0]], a[j[1]], a[j[2]], a[j[3]]])
            F = np.array([F[..., j[0]],
                          F[..., j[1]],
                          F[..., j[2]],
                          F[..., j[3]]])
            j = [0, 1, 2, 3]
            ia = a != a.real
            ni = np.sum(ia) # how many are there

            if ni == 2:
                ji = np.where(ia)

                if a[ji[0]].imag > a[ji[1]].imag:
                    jh = j[ji[0]]
                    j[ji[0]] = j[ji[1]]
                    j[ji[1]] = jh

            elif ni == 4:
                if a[0].imag > a[1].imag:
                    j = np.array([j[1], j[0], j[2], j[3]])

                if a[2].imag > a[3].imag:
                    j = np.array([j[0], j[1], j[3], j[2]])

            alpha = np.array([a[j[0]],
                              a[j[1]],
                              a[j[2]],
                              a[j[3]]])
            
            
            F = np.array([F[..., j[0]],
                          F[..., j[1]],
                          F[..., j[2]],
                          F[..., j[3]]])
            
            # Get power flow direction from poynting vector
            Pd = poynting(F)

            for j in range(4):
                if np.abs(Pd[j]) < INDEX_TOL:
                    Pd[j] = 0

            Pd = np.sign(Pd)

            # j = [3, 0, 1, 2]
            # if (Pd == [-1, -1, 1, 1]) or (Pd == [-1, 0, 0, 1]):
            #     j = [3, 0, 2, 1]

            # alpha = np.array([a[j[0]],
            #                   a[j[1]],
            #                   a[j[2]],
            #                   a[j[3]]])
            
            # F = np.array([F[..., j[0]],
            #               F[..., j[1]],
            #               F[..., j[2]],
            #               F[..., j[3]]])
        Ex = - (exy * F[..., 0] + exz * F[..., 2] + z0 * beta * F[..., 1]) / exx
        Ey = F[..., 0]
        Ez = F[..., 2]

        Hx = beta * F[..., 2] / z0
        Hy = F[..., 3]
        Hz = F[..., 1]

        E = np.array([Ex, Ey, Ez])
        H = np.array([Hx, Hy, Hz])

    return F, alpha, E, H


# def reflect(arg1, arg2, arg3):

#     I = np.identity(4)
#     ms = arg1.shape[-1]

#     # start with the general birefringent layer treatment on line 40
#     Fc = arg1
#     M = arg2
#     Fs = arg3
#     Z0 = z0

#     m = 





        
        

