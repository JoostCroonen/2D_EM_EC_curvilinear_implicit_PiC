import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize
from numpy import sin, cos, tan, tanh, cosh, sqrt, pi
import os

nx, ny = 32, 64
nxc, nyc = nx, ny
nxn, nyn = nxc+1, nyc+1
Lx, Ly = 16, 32
dx, dy = Lx / nxc, Ly / nyc


CnC = False
skewed = False
squared = False
sinus = False
hypertan = False
double_hypertan = False
double_smooth_hs = True

eps = 0.00              # perturbation parameter
tw = 4                  # tanh width parameter
theta = 0               # skew angle
r = 10                   # DSHS density ratio
h = 8                   # DSHS high density region width
s = 5                   # DSHS transition sharpness
L = Ly
b1 = L/4-h/2
b2 = L/4+h/2
b3 = 3*L/4-h/2
b4 = 3*L/4+h/2
o1 = r*b1-0-b1
o2 = r*b2-o1-b2
o3 = r*b3-o2-b3
o4 = r*b4-o3-b4
mv = L+o4

xi_c, eta_c = np.mgrid[dx / 2:Lx - dx / 2:(nxc * 1j), dy / 2:Ly - dy / 2:(nyc * 1j)]
xi_n, eta_n = np.mgrid[0:Lx:(nxn * 1j), 0:Ly:(nyn * 1j)]
xc = xi_c.copy()
yc = eta_c.copy()
xn = xi_n.copy()
yn = eta_n.copy()

def cartesian_to_general_map(x, y):
    '''To convert the particles position from Cartesian geom. to General geom.
    '''
    if CnC:
        xi = x + eps * np.sin(2 * np.pi * x / Lx) * np.sin(2 * np.pi * y / Ly)
        eta = y + eps * np.sin(2 * np.pi * x / Lx) * np.sin(2 * np.pi * y / Ly)
    elif skewed:
        xi = x
        eta = np.cos(theta) * x + np.sin(theta) * y
    elif squared:
        xi = x ** 2
        eta = y ** 2
    elif sinus:
        xi = x + Lx * eps * np.sin(2 * np.pi * x / Lx)
        eta = y
    elif hypertan:
        xi = (Lx - 2 * eps * Lx) / Lx * x - eps * Lx * tanh((Lx / 2 - x)/tw) + eps * Lx
        eta = y
    elif double_hypertan:
        xi =  x
        eta = (Ly-4*eps*Ly)/Ly * y - eps*Ly*tanh(Ly/4 - y) - eps*Ly*tanh(3*Ly/4 - y) + 2*eps*Ly
    elif double_smooth_hs:
        xi = x
        eta = L / mv * (y
                        - y * (0.5 + 0.5 * tanh(s * (y - b1)))
                        + (r * y - o1) * (0.5 + 0.5 * tanh(s * (y - b1)))
                        - (r * y - o1) * (0.5 + 0.5 * tanh(s * (y - b2)))
                        + (y + o2) * (0.5 + 0.5 * tanh(s * (y - b2)))
                        - (y + o2) * (0.5 + 0.5 * tanh(s * (y - b3)))
                        + (r * y - o3) * (0.5 + 0.5 * tanh(s * (y - b3)))
                        - (r * y - o3) * (0.5 + 0.5 * tanh(s * (y - b4)))
                        + (y + o4) * (0.5 + 0.5 * tanh(s * (y - b4))))
    else:
        xi = x
        eta = y
    return xi, eta


def diff_for_inversion(param, target):
    xi, eta = cartesian_to_general_map(param[0], param[1])
    return (xi - target[0]) ** 2 + (eta - target[1]) ** 2


# Function to calculate physical positions of grid nodes
def cart_grid_calculator(xi, eta):
    if xi.shape != eta.shape:
        raise ValueError
    x = np.zeros_like(xi)
    y = np.zeros_like(eta)

    init0 = xi.copy()
    init1 = eta.copy()

    init = np.stack((init0, init1), axis=-1)
    target = np.stack((xi, eta), axis=-1)

    # use this to set bounds on the values of x and y
    bnds = ((None, None), (None, None))
    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            res = minimize(lambda param: diff_for_inversion(param, target[i, j, :]), init[i, j, :], bounds=bnds, tol=1e-14)
            x[i, j] = res.x[0]
            y[i, j] = res.x[1]
    return x, y


def perturbed_inverse_jacobian_elements(x, y):
    if CnC:
        j11 = 1. + 2. * np.pi * eps * np.cos(2. * np.pi * x / Lx) * np.sin(2. * np.pi * y / Ly) / Lx
        j12 = 2. * np.pi * eps * np.sin(2. * np.pi * x / Lx) * np.cos(2. * np.pi * y / Ly) / Ly
        j13 = np.zeros(np.shape(x), np.float64)
        j21 = 2. * np.pi * eps * np.cos(2. * np.pi * x / Lx) * np.sin(2. * np.pi * y / Ly) / Lx
        j22 = 1. + 2. * np.pi * eps * np.sin(2. * np.pi * x / Lx) * np.cos(2. * np.pi * y / Ly) / Ly
        j23 = np.zeros(np.shape(x), np.float64)
        j31 = np.zeros(np.shape(x), np.float64)
        j32 = np.zeros(np.shape(x), np.float64)
        j33 = np.ones(np.shape(x), np.float64)

    elif skewed:
        j11 = np.ones_like(x, np.float64)
        j12 = np.zeros_like(x, np.float64)
        j13 = np.zeros_like(x, np.float64)
        j21 = np.ones_like(x, np.float64)*cos(theta)
        j22 = np.ones_like(x, np.float64)*sin(theta)
        j23 = np.zeros_like(x, np.float64)
        j31 = np.zeros_like(x, np.float64)
        j32 = np.zeros_like(x, np.float64)
        j33 = np.ones_like(x, np.float64)

    #if cylindrical:
    #    j11 = x/sqrt(x**2 + y**2)
    #    j12 = y/sqrt(x**2 + y**2)
    #    j13 = np.zeros_like(x, np.float64)
    #    j21 = -y/(x**2 + y**2)
    #    j22 = x/(x**2 + y**2)
    #    j23 = np.zeros_like(x, np.float64)
    #    j31 = np.zeros_like(x, np.float64)
    #    j32 = np.zeros_like(x, np.float64)
    #    j33 = np.ones_like(x, np.float64)

    elif squared:
        j11 = 2*x
        j12 = np.zeros_like(x, np.float64)
        j13 = np.zeros_like(x, np.float64)
        j21 = np.zeros_like(x, np.float64)
        j22 = 2*y
        j23 = np.zeros_like(x, np.float64)
        j31 = np.zeros_like(x, np.float64)
        j32 = np.zeros_like(x, np.float64)
        j33 = np.ones_like(x, np.float64)

    elif sinus:
        j11 = 1. + 2. * np.pi * eps * np.cos(2. * np.pi * x / Lx) / Lx * Lx
        j12 = np.zeros_like(x, np.float64)
        j13 = np.zeros_like(x, np.float64)
        j21 = np.zeros_like(x, np.float64)
        j22 = np.ones_like(x, np.float64)
        j23 = np.zeros_like(x, np.float64)
        j31 = np.zeros_like(x, np.float64)
        j32 = np.zeros_like(x, np.float64)
        j33 = np.ones_like(x, np.float64)
    elif hypertan:
        j11 = (Lx-2*eps*Lx)/Lx + eps * Lx / (cosh(Lx/2-x)**2)
        j12 = np.zeros_like(x, np.float64)
        j13 = np.zeros_like(x, np.float64)
        j21 = np.zeros_like(x, np.float64)
        j22 = np.ones_like(x, np.float64)
        j23 = np.zeros_like(x, np.float64)
        j31 = np.zeros_like(x, np.float64)
        j32 = np.zeros_like(x, np.float64)
        j33 = np.ones_like(x, np.float64)
    elif double_hypertan:
        j11 = np.ones_like(x, np.float64)
        j12 = np.zeros_like(x, np.float64)
        j13 = np.zeros_like(x, np.float64)
        j21 = np.zeros_like(x, np.float64)
        j22 = (Ly-4*eps*Ly)/Ly + eps * Ly / (cosh(Ly/4-y)**2) + eps * Ly / (cosh(3*Ly/4-y)**2)
        j23 = np.zeros_like(x, np.float64)
        j31 = np.zeros_like(x, np.float64)
        j32 = np.zeros_like(x, np.float64)
        j33 = np.ones_like(x, np.float64)
    if double_smooth_hs:
        j11 = np.ones_like(x, np.float64)
        j12 = np.zeros_like(x, np.float64)
        j13 = np.zeros_like(x, np.float64)
        j21 = np.zeros_like(x, np.float64)
        j22 = L / mv * (1
                        - 0.5 * s * (1 - tanh(s * (y - b1)) ** 2) * y
                        + 0.5 * s * (1 - tanh(s * (y - b1)) ** 2) * (r * y - o1)
                        - 0.5 * s * (1 - tanh(s * (y - b2)) ** 2) * (r * y - o1)
                        + 0.5 * s * (1 - tanh(s * (y - b2)) ** 2) * (y + o2)
                        - 0.5 * s * (1 - tanh(s * (y - b3)) ** 2) * (y + o2)
                        + 0.5 * s * (1 - tanh(s * (y - b3)) ** 2) * (r * y - o3)
                        - 0.5 * s * (1 - tanh(s * (y - b4)) ** 2) * (r * y - o3)
                        + 0.5 * s * (1 - tanh(s * (y - b4)) ** 2) * (y + o4)
                        - 1 * (0.5 + 0.5 * tanh(s * (y - b1)))
                        + r * (0.5 + 0.5 * tanh(s * (y - b1)))
                        - r * (0.5 + 0.5 * tanh(s * (y - b2)))
                        + 1 * (0.5 + 0.5 * tanh(s * (y - b2)))
                        - 1 * (0.5 + 0.5 * tanh(s * (y - b3)))
                        + r * (0.5 + 0.5 * tanh(s * (y - b3)))
                        - r * (0.5 + 0.5 * tanh(s * (y - b4)))
                        + 1 * (0.5 + 0.5 * tanh(s * (y - b4))))
        j23 = np.zeros_like(x, np.float64)
        j31 = np.zeros_like(x, np.float64)
        j32 = np.zeros_like(x, np.float64)
        j33 = np.ones_like(x, np.float64)


    else:
        j11 = np.ones_like(x, np.float64)
        j12 = np.zeros_like(x, np.float64)
        j13 = np.zeros_like(x, np.float64)
        j21 = np.zeros_like(x, np.float64)
        j22 = np.ones_like(x, np.float64)
        j23 = np.zeros_like(x, np.float64)
        j31 = np.zeros_like(x, np.float64)
        j32 = np.zeros_like(x, np.float64)
        j33 = np.ones_like(x, np.float64)
    return j11, j12, j13, j21, j22, j23, j31, j32, j33


def initiate_geometry():
    global xn, yn, xc, yc
    print('Invert grid to find physical grid locations')
    if CnC or skewed or squared or sinus or hypertan or double_hypertan or double_smooth_hs:
        if double_hypertan or double_smooth_hs or sinus or hypertan:
            if double_hypertan:
                a = 'double_tanh_'
                e = str(eps).replace('.', '')
            if double_smooth_hs:
                a = 'double_smooth_hs_'
                e = str(r) + '_' + str(h) + '_' + str(s)
            if sinus:
                a = 'sin_'
                e = str(eps).replace('.', '')
            if hypertan:
                a = 'tanh_'
                e = str(eps).replace('.', '') + '_' + str(tw).replace('.', '')
            name = 'save_data\\grids\\' + a + str(nx) + '_' + str(ny) + '_' + str(Lx).replace('.', '') + '_' + str(Ly).replace('.', '') + '_' + e
            namexc = name + '_xc.npy'
            nameyc = name + '_yc.npy'
            namexn = name + '_xn.npy'
            nameyn = name + '_yn.npy'
            if os.path.exists(namexc):
                xc = np.load(namexc)
                yc = np.load(nameyc)
                xn = np.load(namexn)
                yn = np.load(nameyn)
            else:
                xc, yc = cart_grid_calculator(xi_c, eta_c)
                xn, yn = cart_grid_calculator(xi_n, eta_n)
                np.save(namexc, xc)
                np.save(nameyc, yc)
                np.save(namexn, xn)
                np.save(nameyn, yn)
        else:
            xc, yc = cart_grid_calculator(xi_c, eta_c)
            xn, yn = cart_grid_calculator(xi_n, eta_n)
        dxmin = np.min(xn[1:, :] - xn[0:-1, :])
        dxmax = np.max(xn[1:, :] - xn[0:-1, :])
        dymin = np.min(yn[:, 1:] - yn[:, 0:-1])
        dymax = np.max(yn[:, 1:] - yn[:, 0:-1])
        print(dxmin, dxmax, dxmin/dxmax)
        print(dymin, dymax, dymin/dymax)
    else:
        xc, yc = xi_c, eta_c
        xn, yn = xi_n, eta_n

initiate_geometry()

plt.figure('grid')
plt.gca().set_aspect('equal')
plt.pcolormesh(xn, yn, np.nan*np.ones_like(xc), edgecolors = 'k', linewidth=0.5)
plt.tight_layout()

plt.show()
