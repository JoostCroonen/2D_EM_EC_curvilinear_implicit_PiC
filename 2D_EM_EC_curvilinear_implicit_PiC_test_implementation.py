# 2D Electromagnetic Energy Conserving Curvilinear Implicit PiC Code
# Made by Joost Croonen, Luca Pezini, Fabio Bacchini and Giovanni Lapenta


import numpy as np
from numpy import sin, cos, sqrt, pi, tanh, cosh
from numba import jit, njit, vectorize, guvectorize, float64, int32, uint32, cuda, prange
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy.optimize import newton_krylov, minimize
from concurrent.futures import ThreadPoolExecutor
import time
import datetime
import math
import inspect
import sys
from scipy.special import erf
import os.path

# Used to recompile all the numba functions after the initiation phase. This way all static variables will be correctly initialised in the numba functions.
def recompile_nb_code():
    this_module = sys.modules[__name__]
    module_members = inspect.getmembers(this_module)

    for member_name, member in module_members:
        if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
            member.recompile()


# Simulation parameters
nx, ny = 32, 64
nxc, nyc = nx, ny
nxn, nyn = nxc+1, nyc+1
Lx, Ly = 16, 32
x0, y0 = 0, 0                       # offset of the origin
dx, dy = Lx / nxc, Ly / nyc
invdx, invdy = 1 / dx, 1 / dy
dt = 0.0001                          # time step size
nt = 100                          # number of time steps
ns = 4                              # number of species in plasma
ppcx = 4                            # particles per cell in the x direction
ppcy = 4                            # particles per cell in the x direction
ppc = ppcx * ppcy
npart = ns * ppcx * ppcy * nxc * nyc

# Science case
two_stream = False
weibel = False
thermal = False
drift = False
guassian_tem_5x5 = False
GEM = True

# Geometry presets
CnC = False
skewed = False
squared = False
sinus = False
hypertan = False
double_hypertan = False
double_smooth_hs = True

# Geometry parameters
eps = 0.00               # perturbation parameter
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
if squared:
    x0 = dx
    y0 = dy

# Numeric methods
nearest_neighbour = False
NK_solver_flag = False
np.random.seed(42)

# Data saving and visualization
images = True
save_data = True
save_restart_data = True
frequency_image = 50
frequency_save = 50
frequency_restart = 1000
save_path = 'save_data/temp'
restart_path = 'save_data/temp'

# Restarting
restart = False
restart_time = 3000
restart_location = 'save_data/restart/DSHS_RESTART_160_192_16_5_8_5_4001_0025_2023_06_02_18_26/'

# Iteration parameters
max_iter = 100
atol = 1e-14
rtol = 1e-14

# Validate inputs
science_flags = np.array([thermal, guassian_tem_5x5, two_stream, weibel, GEM])
num_science_flags = np.sum(science_flags)
if num_science_flags > 1:
    raise RuntimeError('Several incompatible science preset flags have been set!')
if num_science_flags+restart > 1:
    raise RuntimeError('Restart flag and science preset flag are set simultaneously')
geometry_flags = np.array([CnC, squared, skewed, sinus, hypertan, double_hypertan])
num_geom_flags = np.sum(geometry_flags)
if num_geom_flags  > 1:
    raise RuntimeError('Several incompatible geometry preset flags have been set!')


# Diagnostics
Ufield = np.ones(nt)
Ub = np.ones(nt)
Ue = np.ones(nt)
Ufc = np.ones(nt)
Uk = np.ones(nt)
Utot = np.ones(nt)
UJdotE = np.ones(nt)
divB = np.ones(nt)
charge = np.ones(nt)
mom = np.ones(nt)
cfl = np.ones(nt)
EdotCurlBAv = np.ones(nt)
BdotCurlEAv = np.ones(nt)
sumDotCurlAv = np.zeros(nt)
deltaU = np.zeros(nt)
E_test = np.ones(nt)
B_test = np.ones(nt)

# Timers
time_tot = 0
time_p2f = 0
time_f2p = 0
time_maxwell = 0
time_init = 0
time_mover = 0
time_newt = 0
time_c2g_part = 0
time_g2c_field = 0
time_c2g_field = 0
time_kry2phys = 0
time_phys2kry = 0
time_linalg = 0
time_energy = 0
time_IO = 0


# Grid
xi_c, eta_c = np.mgrid[x0 + dx / 2:x0 + Lx - dx / 2:(nxc * 1j), y0 + dy / 2:y0 + Ly - dy / 2:(nyc * 1j)]
xi_n, eta_n = np.mgrid[x0 + 0:x0 + Lx:(nxn * 1j), y0 + 0:y0 + Ly:(nyn * 1j)]
xc = xi_c.copy()
yc = eta_c.copy()
xn = xi_n.copy()
yn = eta_n.copy()
dxmin = dx
dxmax = dx
dymin = dy
dymax = dy

# Fields
E1 = np.zeros((nxc, nyc))
E2 = np.zeros((nxc, nyc))
E3 = np.zeros((nxc, nyc))
E1old = np.zeros((nxc, nyc))
E2old = np.zeros((nxc, nyc))
E3old = np.zeros((nxc, nyc))
B1 = np.zeros((nxn, nyn))
B2 = np.zeros((nxn, nyn))
B3 = np.zeros((nxn, nyn))
B1old = np.zeros((nxn, nyn))
B2old = np.zeros((nxn, nyn))
B3old = np.zeros((nxn, nyn))
J1 = np.zeros((nxc, nyc))
J2 = np.zeros((nxc, nyc))
J3 = np.zeros((nxc, nyc))
rho = np.zeros((nxn, nyn))
rhoC = np.zeros((nxc, nyc))
rhomC = np.zeros((nxc, nyc))



# Particles
vth = 0.01
vs = np.array([1, 0])
qs = np.array([-1, 1])
ms = np.array([1, 1])
udrift = 0
vdrift = 0
wdrift = 0
x = np.zeros(npart)
y = np.zeros(npart)
z = np.zeros(npart)
u = np.zeros(npart)
v = np.zeros(npart)
w = np.zeros(npart)
q = np.ones(npart)
qm = np.ones(npart)
m = np.ones(npart)


if GEM:
    x = np.load('save_data/init/GEM_pert_x.npy')
    y = np.load('save_data/init/GEM_pert_y.npy')
    u = np.load('save_data/init/GEM_pert_u.npy')
    v = np.load('save_data/init/GEM_pert_v.npy')
    w = np.load('save_data/init/GEM_pert_w.npy')
    B1 = np.load('save_data/init/GEM_pert_B1.npy')
    B2 = np.load('save_data/init/GEM_pert_B2.npy')
    q = np.load('save_data/init/GEM_pert_q.npy')
    qm = np.load('save_data/init/GEM_pert_qm.npy')
    m = np.load('save_data/init/GEM_pert_m.npy')
    npart = int(np.load('save_data/init/GEM_pert_npart.npy'))
    vth = 0.0

if guassian_tem_5x5:
    B3[nxc//2, nyc//2] = .25/10
    #B3[nxc//2+1, nyc//2] = .125/10
    #B3[nxc//2-1, nyc//2] = .125/10
    #B3[nxc//2, nyc//2+1] = .125/10
    #B3[nxc//2, nyc//2-1] = .125/10
    #B3[nxc//2+1, nyc//2+1] = .0625/10
    #B3[nxc//2+1, nyc//2-1] = .0625/10
    #B3[nxc//2-1, nyc//2+1] = .0625/10
    #B3[nxc//2-1, nyc//2-1] = .0625/10

if thermal:
    vth = 0.01
    udrift = 0
    vdrift = 0
    wdrift = 0

if two_stream:
    vth = 0.001
    udrift = .2
    vdrift = 0
    wdrift = 0

if weibel:
    vth = 0.001
    udrift = 0
    vdrift = .2
    wdrift = 0

if restart:
    x = np.load(restart_location + 'x_' + str(restart_time).zfill(5) + '.npy')
    y = np.load(restart_location + 'y_' + str(restart_time).zfill(5) + '.npy')
    u = np.load(restart_location + 'u_' + str(restart_time).zfill(5) + '.npy')
    v = np.load(restart_location + 'v_' + str(restart_time).zfill(5) + '.npy')
    w = np.load(restart_location + 'w_' + str(restart_time).zfill(5) + '.npy')
    B1 = np.load(restart_location + 'B1_' + str(restart_time).zfill(5) + '.npy')
    B2 = np.load(restart_location + 'B2_' + str(restart_time).zfill(5) + '.npy')
    B3 = np.load(restart_location + 'B3_' + str(restart_time).zfill(5) + '.npy')
    E1 = np.load(restart_location + 'E1_' + str(restart_time).zfill(5) + '.npy')
    E2 = np.load(restart_location + 'E2_' + str(restart_time).zfill(5) + '.npy')
    E3 = np.load(restart_location + 'E3_' + str(restart_time).zfill(5) + '.npy')
    q = np.load(restart_location + 'q_' + str(restart_time).zfill(5) + '.npy')
    m = np.load(restart_location + 'm_' + str(restart_time).zfill(5) + '.npy')
    qm = np.load(restart_location + 'qm_' + str(restart_time).zfill(5) + '.npy')
    npart = x.size
    vth = 0.0

# Geometry
Jn = np.ones((nxn, nyn))
Jc = np.ones((nxc, nyc))
invJn = 1 / Jn
invJc = 1 / Jc

g11n = np.ones((nxn, nyn))
g12n = np.zeros((nxn, nyn))
g13n = np.zeros((nxn, nyn))
g21n = np.zeros((nxn, nyn))
g22n = np.ones((nxn, nyn))
g23n = np.zeros((nxn, nyn))
g31n = np.zeros((nxn, nyn))
g32n = np.zeros((nxn, nyn))
g33n = np.ones((nxn, nyn))
g11c = np.ones((nxc, nyc))
g12c = np.zeros((nxc, nyc))
g13c = np.zeros((nxc, nyc))
g21c = np.zeros((nxc, nyc))
g22c = np.ones((nxc, nyc))
g23c = np.zeros((nxc, nyc))
g31c = np.zeros((nxc, nyc))
g32c = np.zeros((nxc, nyc))
g33c = np.ones((nxc, nyc))

J11n = np.ones((nxn, nyn))
J12n = np.zeros((nxn, nyn))
J13n = np.zeros((nxn, nyn))
J21n = np.zeros((nxn, nyn))
J22n = np.ones((nxn, nyn))
J23n = np.zeros((nxn, nyn))
J31n = np.zeros((nxn, nyn))
J32n = np.zeros((nxn, nyn))
J33n = np.ones((nxn, nyn))
J11c = np.ones((nxc, nyc))
J12c = np.zeros((nxc, nyc))
J13c = np.zeros((nxc, nyc))
J21c = np.zeros((nxc, nyc))
J22c = np.ones((nxc, nyc))
J23c = np.zeros((nxc, nyc))
J31c = np.zeros((nxc, nyc))
J32c = np.zeros((nxc, nyc))
J33c = np.ones((nxc, nyc))

j11n = np.ones((nxn, nyn))
j12n = np.zeros((nxn, nyn))
j13n = np.zeros((nxn, nyn))
j21n = np.zeros((nxn, nyn))
j22n = np.ones((nxn, nyn))
j23n = np.zeros((nxn, nyn))
j31n = np.zeros((nxn, nyn))
j32n = np.zeros((nxn, nyn))
j33n = np.ones((nxn, nyn))
j11c = np.ones((nxc, nyc))
j12c = np.zeros((nxc, nyc))
j13c = np.zeros((nxc, nyc))
j21c = np.zeros((nxc, nyc))
j22c = np.ones((nxc, nyc))
j23c = np.zeros((nxc, nyc))
j31c = np.zeros((nxc, nyc))
j32c = np.zeros((nxc, nyc))
j33c = np.ones((nxc, nyc))



@njit('f8[:,:](f8[:,:])')
def ddx_n2c(A):
    return ((A[1:, :-1] + A[1:, 1:]) * .5 - (A[:-1, :-1] + A[:-1, 1:]) * .5) * invdx

@njit('f8[:,:](f8[:,:])')
def ddy_n2c(A):
    return ((A[:-1, 1:] + A[1:, 1:]) * .5 - (A[:-1, :-1] + A[1:, :-1]) * .5) * invdy

@njit('f8[:,:](f8[:,:])')
def ddx_c2n(A):
    A_extended = np.zeros((A.shape[0] + 2, A.shape[1] + 2))
    A_extended[1:-1, 1:-1] = A
    A_extended[0, 0] = A[-1, -1]
    A_extended[0, -1] = A[-1, 0]
    A_extended[-1, 0] = A[0, -1]
    A_extended[-1, -1] = A[0, 0]
    A_extended[0, 1:-1] = A[-1, :]
    A_extended[-1, 1:-1] = A[0, :]
    A_extended[1:-1, 0] = A[:, -1]
    A_extended[1:-1, -1] = A[:, 0]
    return ((A_extended[1:, :-1] + A_extended[1:, 1:]) * .5 - (A_extended[:-1, :-1] + A_extended[:-1, 1:]) * .5) * invdx

@njit('f8[:,:](f8[:,:])')
def ddy_c2n(A):
    A_extended = np.zeros((A.shape[0] + 2, A.shape[1] + 2))
    A_extended[1:-1, 1:-1] = A
    A_extended[0, 0] = A[-1, -1]
    A_extended[0, -1] = A[-1, 0]
    A_extended[-1, 0] = A[0, -1]
    A_extended[-1, -1] = A[0, 0]
    A_extended[0, 1:-1] = A[-1, :]
    A_extended[-1, 1:-1] = A[0, :]
    A_extended[1:-1, 0] = A[:, -1]
    A_extended[1:-1, -1] = A[:, 0]
    return ((A_extended[:-1, 1:] + A_extended[1:, 1:]) * .5 - (A_extended[:-1, :-1] + A_extended[1:, :-1]) * .5) * invdy


@njit('UniTuple(f8[:,:], 3)(f8[:,:], f8[:,:], f8[:,:])')
def curlE(A1, A2, A3):
    curl1 = invJn * (ddy_c2n(g31c * A1 + g32c * A2 + g33c * A3))
    curl2 = invJn * (-ddx_c2n(g31c * A1 + g32c * A2 + g33c * A3))
    curl3 = invJn * (ddx_c2n(g21c * A1 + g22c * A2 + g23c * A3) - ddy_c2n(g11c * A1 + g12c * A2 + g13c * A3))
    return curl1, curl2, curl3

# numba njit doenst work with global variables. to resolve this we need to add all global variables as variables on function call. or if the globals are static, then they need to be set before the function is compiled such that they have the correct value
@njit('UniTuple(f8[:,:], 3)(f8[:,:], f8[:,:], f8[:,:])')
def curlB(A1, A2, A3):
    curl1 = invJc * (ddy_n2c(g31n * A1 + g32n * A2 + g33n * A3))
    curl2 = invJc * (-ddx_n2c(g31n * A1 + g32n * A2 + g33n * A3))
    curl3 = invJc * (ddx_n2c(g21n * A1 + g22n * A2 + g23n * A3) - ddy_n2c(g11n * A1 + g12n * A2 + g13n * A3))
    return curl1, curl2, curl3


@njit("f8[:](f8[:], f8[:], f8[:,:])")
def grid_to_particle(x, y, A):
    Ap = np.zeros_like(x)
    xsize = A.shape[0]
    ysize = A.shape[1]
    #invJ = np.zeros_like(A)
    #if xsize == nxc:
    #    invJ = invJc
    #elif xsize == nxn:
    #    invJ = invJn
    #else:
    #    print('this shouldnt happen')
    for i in range(npart):
        xa = (x[i] - (nxn - xsize) * (dx * .5)) * invdx
        ya = (y[i] - (nyn - ysize) * (dy * .5)) * invdy
        i1 = int(np.floor(xa))
        i2 = i1 + 1
        j1 = int(np.floor(ya))
        j2 = j1 + 1
        wx2 = xa - i1 #np.float64(i1)
        wx1 = 1.0 - wx2
        wy2 = ya - j1 #np.float64(j1)
        wy1 = 1.0 - wy2
        #i1, i2 = i1 % xsize, i2 % xsize
        #j1, j2 = j1 % ysize, j2 % ysize
        if nearest_neighbour:
            if wx1 >= wx2:
                wx1 = 1.0
                wx2 = 0.0
            elif wx1 < wx2:
                wx1 = 0.0
                wx2 = 1.0
            if wy1 >= wy2:
                wy1 = 1.0
                wy2 = 0.0
            elif wy1 < wy2:
                wy1 = 0.0
                wy2 = 1.0
        while i1 >= xsize:
            i1 -= xsize
        while i2 >= xsize:
            i2 -= xsize
        while j1 >= ysize:
            j1 -= ysize
        while j2 >= ysize:
            j2 -= ysize
        while i1 < 0:
            i1 += xsize
        while i2 < 0:
            i2 += xsize
        while j1 < 0:
            j1 += ysize
        while j2 < 0:
            j2 += ysize
        Ap[i] += wx1 * wy1 * A[i1, j1]
        Ap[i] += wx2 * wy1 * A[i2, j1]
        Ap[i] += wx1 * wy2 * A[i1, j2]
        Ap[i] += wx2 * wy2 * A[i2, j2]
    return Ap


@njit("f8[:,:](f8[:], f8[:], f8[:])")
def particle_to_grid_rhoN(x, y, q):
    ''' Interpolation particle to grid - charge rho -> n
    '''

    rho = np.zeros((nxn, nyn), np.float64)

    for i in range(npart):
        xa = (x[i]) * invdx
        ya = (y[i]) * invdy
        i1 = int(np.floor(xa))
        i2 = i1 + 1
        j1 = int(np.floor(ya))
        j2 = j1 + 1
        wx2 = xa - np.float64(i1)
        wx1 = 1.0 - wx2
        wy2 = ya - np.float64(j1)
        wy1 = 1.0 - wy2
        #i1, i2 = i1 % nxn, i2 % nxn
        #j1, j2 = j1 % nyn, j2 % nyn
        while i1 >= nxn:
            i1 -= nxn
        while i2 >= nxn:
            i2 -= nxn
        while j1 >= nyn:
            j1 -= nyn
        while j2 >= nyn:
            j2 -= nyn
        while i1 < 0:
            i1 += nxn
        while i2 < 0:
            i2 += nxn
        while j1 < 0:
            j1 += nyn
        while j2 < 0:
            j2 += nyn

        rho[i1, j1] += wx1 * wy1 * q[i] * invdx * invdy
        rho[i2, j1] += wx2 * wy1 * q[i] * invdx * invdy
        rho[i1, j2] += wx1 * wy2 * q[i] * invdx * invdy
        rho[i2, j2] += wx2 * wy2 * q[i] * invdx * invdy

    return rho


@njit("UniTuple(f8[:,:], 2)(f8[:], f8[:], f8[:])")
def particle_to_grid_rhoC(x, y, q):
    ''' Interpolation particle to grid - charge rho -> n
    '''

    rho = np.zeros((nxc, nyc), np.float64)
    rhom = np.zeros((nxc, nyc), np.float64)

    for i in range(npart):
        xa = (x[i]) * invdx
        ya = (y[i]) * invdy
        i1 = int(np.floor(xa))
        i2 = i1 + 1
        j1 = int(np.floor(ya))
        j2 = j1 + 1
        wx2 = xa - np.float64(i1)
        wx1 = 1.0 - wx2
        wy2 = ya - np.float64(j1)
        wy1 = 1.0 - wy2
        #i1, i2 = i1 % nxn, i2 % nxn
        #j1, j2 = j1 % nyn, j2 % nyn
        while i1 >= nxc:
            i1 -= nxc
        while i2 >= nxc:
            i2 -= nxc
        while j1 >= nyc:
            j1 -= nyc
        while j2 >= nyc:
            j2 -= nyc
        while i1 < 0:
            i1 += nxc
        while i2 < 0:
            i2 += nxc
        while j1 < 0:
            j1 += nyc
        while j2 < 0:
            j2 += nyc

        rho[i1, j1] += wx1 * wy1 * q[i] * invdx * invdy
        rho[i2, j1] += wx2 * wy1 * q[i] * invdx * invdy
        rho[i1, j2] += wx1 * wy2 * q[i] * invdx * invdy
        rho[i2, j2] += wx2 * wy2 * q[i] * invdx * invdy

        rhom[i1, j1] += wx1 * wy1 * qm[i] * invdx * invdy
        rhom[i2, j1] += wx2 * wy1 * qm[i] * invdx * invdy
        rhom[i1, j2] += wx1 * wy2 * qm[i] * invdx * invdy
        rhom[i2, j2] += wx2 * wy2 * qm[i] * invdx * invdy

    return rho, rhom


@njit("UniTuple(f8[:,:], 3)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])")
def particle_to_grid_J(xk, yk, uk, vk, wk, q):
    ''' Interpolation particle to grid - current J -> c
    '''

    Jx = np.zeros((nxc, nyc), np.float64)
    Jy = np.zeros((nxc, nyc), np.float64)
    Jz = np.zeros((nxc, nyc), np.float64)

    for i in range(npart):
        xa = (xk[i] - dx * .5) * invdx
        ya = (yk[i] - dy * .5) * invdy
        i1 = int(np.floor(xa))
        i2 = i1 + 1
        j1 = int(np.floor(ya))
        j2 = j1 + 1
        wx2 = xa - np.float64(i1)
        wx1 = 1.0 - wx2
        wy2 = ya - np.float64(j1)
        wy1 = 1.0 - wy2
        #i1, i2 = i1 % nxc, i2 % nxc
        #j1, j2 = j1 % nyc, j2 % nyc
        if nearest_neighbour:
            if wx1 >= wx2:
                wx1 = 1.0
                wx2 = 0.0
            elif wx1 < wx2:
                wx1 = 0.0
                wx2 = 1.0
            if wy1 >= wy2:
                wy1 = 1.0
                wy2 = 0.0
            elif wy1 < wy2:
                wy1 = 0.0
                wy2 = 1.0
        while i1 >= nxc:
            i1 -= nxc
        while i2 >= nxc:
            i2 -= nxc
        while j1 >= nyc:
            j1 -= nyc
        while j2 >= nyc:
            j2 -= nyc
        while i1 < 0:
            i1 += nxc
        while i2 < 0:
            i2 += nxc
        while j1 < 0:
            j1 += nyc
        while j2 < 0:
            j2 += nyc

        Jx[i1, j1] += wx1 * wy1 * q[i] * uk[i] * invdx * invdy * invJc[i1, j1]
        Jy[i1, j1] += wx1 * wy1 * q[i] * vk[i] * invdx * invdy * invJc[i1, j1]
        Jz[i1, j1] += wx1 * wy1 * q[i] * wk[i] * invdx * invdy * invJc[i1, j1]
        Jx[i2, j1] += wx2 * wy1 * q[i] * uk[i] * invdx * invdy * invJc[i2, j1]
        Jy[i2, j1] += wx2 * wy1 * q[i] * vk[i] * invdx * invdy * invJc[i2, j1]
        Jz[i2, j1] += wx2 * wy1 * q[i] * wk[i] * invdx * invdy * invJc[i2, j1]
        Jx[i1, j2] += wx1 * wy2 * q[i] * uk[i] * invdx * invdy * invJc[i1, j2]
        Jy[i1, j2] += wx1 * wy2 * q[i] * vk[i] * invdx * invdy * invJc[i1, j2]
        Jz[i1, j2] += wx1 * wy2 * q[i] * wk[i] * invdx * invdy * invJc[i1, j2]
        Jx[i2, j2] += wx2 * wy2 * q[i] * uk[i] * invdx * invdy * invJc[i2, j2]
        Jy[i2, j2] += wx2 * wy2 * q[i] * vk[i] * invdx * invdy * invJc[i2, j2]
        Jz[i2, j2] += wx2 * wy2 * q[i] * wk[i] * invdx * invdy * invJc[i2, j2]

    return Jx, Jy, Jz


@njit('UniTuple(f8[:,:], 3)(f8[:,:], f8[:,:], f8[:,:])')
def cartesian_to_general_fieldC(Ax, Ay, Az):
    A1 = j11c * Ax + j12c * Ay + j13c * Az
    A2 = j21c * Ax + j22c * Ay + j23c * Az
    A3 = j31c * Ax + j32c * Ay + j33c * Az
    return A1, A2, A3


@njit('UniTuple(f8[:,:], 3)(f8[:,:], f8[:,:], f8[:,:])')
def cartesian_to_general_fieldN(Ax, Ay, Az):
    A1 = j11n * Ax + j12n * Ay + j13n * Az
    A2 = j21n * Ax + j22n * Ay + j23n * Az
    A3 = j31n * Ax + j32n * Ay + j33n * Az
    return A1, A2, A3


@njit('UniTuple(f8[:,:], 3)(f8[:,:], f8[:,:], f8[:,:])')
def general_to_cartesian_fieldC(A1, A2, A3):
    Ax = J11c * A1 + J12c * A2 + J13c * A3
    Ay = J21c * A1 + J22c * A2 + J23c * A3
    Az = J31c * A1 + J32c * A2 + J33c * A3
    return Ax, Ay, Az


@njit('UniTuple(f8[:,:], 3)(f8[:,:], f8[:,:], f8[:,:])')
def general_to_cartesian_fieldN(A1, A2, A3):
    Ax = J11n * A1 + J12n * A2 + J13n * A3
    Ay = J21n * A1 + J22n * A2 + J23n * A3
    Az = J31n * A1 + J32n * A2 + J33n * A3
    return Ax, Ay, Az


@njit('UniTuple(f8[:], 2)(f8[:], f8[:])')
def cartesian_to_general_part(x, y):
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
        eta = (Ly-4*eps*Ly)/Ly * y - eps*Ly*np.tanh(Ly/4 - y) - eps*Ly*np.tanh(3*Ly/4 - y) + 2*eps*Ly
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


#@njit('f8[:](f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:], f8[:])')
def phys_to_krylov(E1k, E2k, E3k, uk, vk, wk):
    ''' To populate the Krylov vector using physiscs vectors
    E1,E2,E3 are 2D arrays
    u,v,w of dimensions npart
    '''
    global nxc, nyc, nxn, nyn, npart, time_phys2kry

    #start = time.time()
    ykrylov = np.zeros(3 * nxc * nyc + 3 * npart, np.float64)

    ykrylov[0:nxc*nyc] = np.reshape(E1k, (nxc*nyc))
    ykrylov[nxc*nyc:2*nxc*nyc] = np.reshape(E2k, (nxc*nyc))
    ykrylov[2*nxc*nyc:3*nxc*nyc] = np.reshape(E3k, (nxc*nyc))
    ykrylov[3*nxc*nyc:3*nxc*nyc + npart] = uk
    ykrylov[3*nxc*nyc + npart:3*nxc*nyc + 2*npart] = vk
    ykrylov[3*nxc*nyc + 2*npart:3*nxc*nyc + 3*npart] = wk
    #time_phys2kry += time.time() - start

    return ykrylov


#@njit('Tuple((f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:], f8[:]))(f8[:])')
def krylov_to_phys(xkrylov):
    ''' To populate the physiscs vectors using the Krylov space vector
    E1,E2,E3 are 2D arrays of dimension (nx,ny)
    unew,vnew,wnew of dimensions npart1+npart2
    '''
    #global nx, ny, npart, time_kry2phys

    #start = time.time()
    E1k = np.reshape(xkrylov[0:nxc*nyc].copy(), (nxc, nyc))
    E2k = np.reshape(xkrylov[nxc*nyc:2*nxc*nyc].copy(), (nxc, nyc))
    E3k = np.reshape(xkrylov[2*nxc*nyc:3*nxc*nyc].copy(), (nxc, nyc))
    uk = xkrylov[3*nxc*nyc:3*nxc*nyc + npart]
    vk = xkrylov[3*nxc*nyc + npart:3*nxc*nyc + 2*npart]
    wk = xkrylov[3*nxc*nyc + 2*npart:3*nxc*nyc + 3*npart]
    #time_kry2phys += time.time() - start

    return E1k, E2k, E3k, uk, vk, wk


def newton_krylov_iterator(NK):
    global E1, E2, E3, u, v, w, time_linalg, time_phys2kry
    if NK:
        start = time.time()
        guess = phys_to_krylov(E1, E2, E3, u, v, w)
        time_phys2kry += time.time() - start
        sol = newton_krylov(residual, guess, method='lgmres', verbose=1, f_tol=atol, f_rtol=rtol)
        print('Residual: %g' % abs(residual(sol)).max())
        if abs(residual(sol)).max()>1e-14:
            raise RuntimeError('Failed to converge')
    else:
        start = time.time()
        guess = phys_to_krylov(E1, E2, E3, u, v, w)
        time_phys2kry += time.time() - start
        err = 1.
        rerr = 1.
        k = 0
        xkrylov = guess
        #while not((err < atol) and (rerr < rtol)) and k <= max_iter:
        while (err > atol) or (rerr > rtol):
            k += 1
            xkold = xkrylov
            xkrylov = xkrylov - residual(xkrylov)
            start = time.time()
            err = np.linalg.norm(xkrylov - xkold)
            rerr = np.linalg.norm((xkrylov - xkold)/np.linalg.norm(xkrylov))
            time_linalg += time.time() - start
            print(k, err)
            if k == max_iter:
                raise RuntimeError('Failed to converge')
        sol = xkrylov
    return sol


@njit('UniTuple(f8[:],2)(f8[:],f8[:])')
def boundary_particles(x, y):
    for i in range(x.shape[0]):
        if x[i] > Lx:
            x[i] -= Lx
        if x[i] < 0:
            x[i] += Lx
    for j in range(y.shape[0]):
        if y[j] > Ly:
            y[j] -= Ly
        if y[j] < 0:
            y[j] += Ly
    return x, y


def residual(krylov):
    global E1, E2, E3, B1, B2, B3, x, y, u, v, w, q, qm, dt, time_p2f, time_f2p, time_maxwell,  time_mover, time_newt, time_c2g_part, time_g2c_field, time_c2g_field, time_kry2phys, time_phys2kry

    # First guess for E and v from NK iterator
    start = time.time()
    E1new, E2new, E3new, unew, vnew, wnew = krylov_to_phys(krylov)
    time_kry2phys += time.time() - start

    ubar = (unew + u) * .5
    vbar = (vnew + v) * .5
    wbar = (wnew + w) * .5

    # Advance particles
    start = time.time()
    xbar = x + ubar * dt * .5
    ybar = y + vbar * dt * .5

    # Periodic boundaries for particles
    xbar, ybar = boundary_particles(xbar, ybar)
    time_mover += time.time() - start

    # Move particles to general grid
    start = time.time()
    xgenbar, ygenbar = cartesian_to_general_part(xbar, ybar)
    time_c2g_part += time.time() - start

    # Interpolate current on grid from particles
    start = time.time()
    Jx, Jy, Jz = particle_to_grid_J(xgenbar, ygenbar, ubar, vbar, wbar, q)
    time_p2f += time.time() - start

    # Convert from xyz-current to xi-eta-zeta-current
    start = time.time()
    J1, J2, J3 = cartesian_to_general_fieldC(Jx, Jy, Jz)
    time_c2g_field += time.time() - start

    start = time.time()
    # Calculate curl of E
    E1bar = (E1new + E1) * .5
    E2bar = (E2new + E2) * .5
    E3bar = (E3new + E3) * .5
    curlE1, curlE2, curlE3 = curlE(E1bar, E2bar, E3bar)

    # Update B using Maxwell equation
    B1bar = B1 - dt * curlE1 * .5
    B2bar = B2 - dt * curlE2 * .5
    B3bar = B3 - dt * curlE3 * .5

    # Calculate curl of B
    curlB1, curlB2, curlB3 = curlB(B1bar, B2bar, B3bar)

    # Update E using Maxwell equation
    E1max = E1 + dt * curlB1 - 4 * pi * dt * J1
    E2max = E2 + dt * curlB2 - 4 * pi * dt * J2
    E3max = E3 + dt * curlB3 - 4 * pi * dt * J3

    time_maxwell += time.time() - start

    # Calculate residual of E by subtracting Emax from Enew
    resE1 = E1new - E1max
    resE2 = E2new - E2max
    resE3 = E3new - E3max

    # Convert fields from general to cartesian coordinates
    start = time.time()
    Ex, Ey, Ez = general_to_cartesian_fieldC(E1bar, E2bar, E3bar)
    Bx, By, Bz = general_to_cartesian_fieldN(B1bar, B2bar, B3bar)
    time_g2c_field += time.time() - start

    # Interpolate fields to particles
    start = time.time()
    Exp = grid_to_particle(xgenbar, ygenbar, Ex)
    Eyp = grid_to_particle(xgenbar, ygenbar, Ey)
    Ezp = grid_to_particle(xgenbar, ygenbar, Ez)
    Bxp = grid_to_particle(xgenbar, ygenbar, Bx)
    Byp = grid_to_particle(xgenbar, ygenbar, By)
    Bzp = grid_to_particle(xgenbar, ygenbar, Bz)
    time_f2p += time.time() - start

    # Update velocity using newtons equation
    start = time.time()
    unewt = u + qm * (Exp + vbar * Bzp - wbar * Byp) * dt
    vnewt = v + qm * (Eyp - ubar * Bzp + wbar * Bxp) * dt
    wnewt = w + qm * (Ezp + ubar * Byp - vbar * Bxp) * dt
    time_newt += time.time() - start

    # Calculate residual of v by subtracting vnewt from vnew
    resu = unew - unewt
    resv = vnew - vnewt
    resw = wnew - wnewt

    start = time.time()
    res_krylov = phys_to_krylov(resE1, resE2, resE3, resu, resv, resw)
    time_phys2kry += time.time() - start
    return res_krylov


def maxwell_solver(sol):
    global E1, E2, E3, B1, B2, B3, J1, J2, J3, x, y, u, v, w, rhoC, rhomC, time_mover, time_maxwell, time_c2g_part, time_p2f, time_c2g_field, time_phys2kry, time_kry2phys
    start = time.time()
    E1new, E2new, E3new, unew, vnew, wnew = krylov_to_phys(sol)
    time_kry2phys += time.time() - start

    start = time.time()

    ubar = (unew + u) * .5
    vbar = (vnew + v) * .5
    wbar = (wnew + w) * .5
    xnew = x + ubar * dt
    ynew = y + vbar * dt

    xnew, ynew = boundary_particles(xnew, ynew)
    time_mover += time.time() - start

    start = time.time()
    E1bar = (E1new + E1) / 2.
    E2bar = (E2new + E2) / 2.
    E3bar = (E3new + E3) / 2.

    curlE1, curlE2, curlE3 = curlE(E1bar, E2bar, E3bar)

    B1new = B1 - dt * curlE1
    B2new = B2 - dt * curlE2
    B3new = B3 - dt * curlE3
    time_maxwell += time.time() - start

    start = time.time()
    xgen, ygen = cartesian_to_general_part(xnew, ynew)
    time_c2g_part += time.time() - start
    start = time.time()
    Jx, Jy, Jz = particle_to_grid_J(xgen, ygen, unew, vnew, wnew, q)
    time_p2f += time.time() - start
    start = time.time()
    J1new, J2new, J3new = cartesian_to_general_fieldC(Jx, Jy, Jz)
    time_c2g_field += time.time() - start

    start = time.time()
    time_p2f += time.time() - start

    x = xnew
    y = ynew
    u = unew
    v = vnew
    w = wnew
    E1old = E1
    E2old = E2
    E3old = E3
    E1 = E1new
    E2 = E2new
    E3 = E3new
    B1old = B1
    B2old = B2
    B3old = B3
    B1 = B1new
    B2 = B2new
    B3 = B3new
    J1 = J1new
    J2 = J2new
    J3 = J3new
    return


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
        eta = (Ly-4*eps*Ly)/Ly * y - eps*Ly*np.tanh(Ly/4 - y) - eps*Ly*np.tanh(3*Ly/4 - y) + 2*eps*Ly
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
        j11 = (Lx-2*eps*Lx)/Lx + eps * Lx / (np.cosh((Lx/2-x)/tw)**2) / tw
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
        j22 = (Ly-4*eps*Ly)/Ly + eps * Ly / (np.cosh(Ly/4-y)**2) + eps * Ly / (np.cosh(3*Ly/4-y)**2)
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
        print(dxmin, dxmax)
        print(dymin, dymax)
    else:
        xc, yc = xi_c, eta_c
        xn, yn = xi_n, eta_n
    print('Build geometry for grid centers')
    for i in range(nxc):
        for j in range(nyc):
            j11c[i, j], j12c[i, j], j13c[i, j], j21c[i, j], j22c[i, j], j23c[i, j], j31c[i, j], j32c[i, j], j33c[i, j] = perturbed_inverse_jacobian_elements(xc[i, j], yc[i, j])
            inverse_jacobian_C = np.array([[j11c[i, j], j12c[i, j], j13c[i, j]], [j21c[i, j], j22c[i, j], j23c[i, j]], [j31c[i, j], j32c[i, j], j33c[i, j]]])
            jacobian_C = np.linalg.inv(inverse_jacobian_C)
            invJc[i, j] = np.linalg.det(inverse_jacobian_C)
            Jc[i, j] = np.linalg.det(jacobian_C)
            J11c[i, j] = jacobian_C[0, 0]
            J21c[i, j] = jacobian_C[1, 0]
            J31c[i, j] = jacobian_C[2, 0]
            J12c[i, j] = jacobian_C[0, 1]
            J22c[i, j] = jacobian_C[1, 1]
            J32c[i, j] = jacobian_C[2, 1]
            J13c[i, j] = jacobian_C[0, 2]
            J23c[i, j] = jacobian_C[1, 2]
            J33c[i, j] = jacobian_C[2, 2]
            g11c[i, j] = jacobian_C[0, 0] * jacobian_C[0, 0] + jacobian_C[1, 0] * jacobian_C[1, 0] + jacobian_C[2, 0] * jacobian_C[2, 0]
            g21c[i, j] = jacobian_C[0, 0] * jacobian_C[0, 1] + jacobian_C[1, 0] * jacobian_C[1, 1] + jacobian_C[2, 0] * jacobian_C[2, 1]
            g31c[i, j] = jacobian_C[0, 0] * jacobian_C[0, 2] + jacobian_C[1, 0] * jacobian_C[1, 2] + jacobian_C[2, 0] * jacobian_C[2, 2]
            g12c[i, j] = jacobian_C[0, 0] * jacobian_C[0, 1] + jacobian_C[1, 0] * jacobian_C[1, 1] + jacobian_C[2, 0] * jacobian_C[2, 1]
            g22c[i, j] = jacobian_C[0, 1] * jacobian_C[0, 1] + jacobian_C[1, 1] * jacobian_C[1, 1] + jacobian_C[2, 1] * jacobian_C[2, 1]
            g32c[i, j] = jacobian_C[0, 1] * jacobian_C[0, 2] + jacobian_C[1, 1] * jacobian_C[1, 2] + jacobian_C[2, 1] * jacobian_C[2, 2]
            g13c[i, j] = jacobian_C[0, 0] * jacobian_C[0, 2] + jacobian_C[1, 0] * jacobian_C[1, 2] + jacobian_C[2, 0] * jacobian_C[2, 2]
            g23c[i, j] = jacobian_C[0, 1] * jacobian_C[0, 2] + jacobian_C[1, 1] * jacobian_C[1, 2] + jacobian_C[2, 1] * jacobian_C[2, 2]
            g33c[i, j] = jacobian_C[0, 2] * jacobian_C[0, 2] + jacobian_C[1, 2] * jacobian_C[1, 2] + jacobian_C[2, 2] * jacobian_C[2, 2]
    print('Build geometry for grid nodes')
    for i in range(nxn):
        for j in range(nyn):
            j11n[i, j], j12n[i, j], j13n[i, j], j21n[i, j], j22n[i, j], j23n[i, j], j31n[i, j], j32n[i, j], j33n[i, j] = perturbed_inverse_jacobian_elements(xn[i, j], yn[i, j])
            inverse_jacobian_N = np.array([[j11n[i, j], j12n[i, j], j13n[i, j]], [j21n[i, j], j22n[i, j], j23n[i, j]], [j31n[i, j], j32n[i, j], j33n[i, j]]])
            jacobian_N = np.linalg.inv(inverse_jacobian_N)
            invJn[i, j] = np.linalg.det(inverse_jacobian_N)
            Jn[i, j] = np.linalg.det(jacobian_N)
            J11n[i, j] = jacobian_N[0, 0]
            J21n[i, j] = jacobian_N[1, 0]
            J31n[i, j] = jacobian_N[2, 0]
            J12n[i, j] = jacobian_N[0, 1]
            J22n[i, j] = jacobian_N[1, 1]
            J32n[i, j] = jacobian_N[2, 1]
            J13n[i, j] = jacobian_N[0, 2]
            J23n[i, j] = jacobian_N[1, 2]
            J33n[i, j] = jacobian_N[2, 2]
            g11n[i, j] = jacobian_N[0, 0] * jacobian_N[0, 0] + jacobian_N[1, 0] * jacobian_N[1, 0] + jacobian_N[2, 0] * jacobian_N[2, 0]
            g21n[i, j] = jacobian_N[0, 0] * jacobian_N[0, 1] + jacobian_N[1, 0] * jacobian_N[1, 1] + jacobian_N[2, 0] * jacobian_N[2, 1]
            g31n[i, j] = jacobian_N[0, 0] * jacobian_N[0, 2] + jacobian_N[1, 0] * jacobian_N[1, 2] + jacobian_N[2, 0] * jacobian_N[2, 2]
            g12n[i, j] = jacobian_N[0, 0] * jacobian_N[0, 1] + jacobian_N[1, 0] * jacobian_N[1, 1] + jacobian_N[2, 0] * jacobian_N[2, 1]
            g22n[i, j] = jacobian_N[0, 1] * jacobian_N[0, 1] + jacobian_N[1, 1] * jacobian_N[1, 1] + jacobian_N[2, 1] * jacobian_N[2, 1]
            g32n[i, j] = jacobian_N[0, 1] * jacobian_N[0, 2] + jacobian_N[1, 1] * jacobian_N[1, 2] + jacobian_N[2, 1] * jacobian_N[2, 2]
            g13n[i, j] = jacobian_N[0, 0] * jacobian_N[0, 2] + jacobian_N[1, 0] * jacobian_N[1, 2] + jacobian_N[2, 0] * jacobian_N[2, 2]
            g23n[i, j] = jacobian_N[0, 1] * jacobian_N[0, 2] + jacobian_N[1, 1] * jacobian_N[1, 2] + jacobian_N[2, 1] * jacobian_N[2, 2]
            g33n[i, j] = jacobian_N[0, 2] * jacobian_N[0, 2] + jacobian_N[1, 2] * jacobian_N[1, 2] + jacobian_N[2, 2] * jacobian_N[2, 2]

            invJn[-1, j] = invJn[0, j]
            Jn[-1, j]    = Jn[0, j]
            J11n[-1, j] = J11n[0, j]
            J21n[-1, j] = J21n[0, j]
            J31n[-1, j] = J31n[0, j]
            J12n[-1, j] = J12n[0, j]
            J22n[-1, j] = J22n[0, j]
            J32n[-1, j] = J32n[0, j]
            J13n[-1, j] = J13n[0, j]
            J23n[-1, j] = J23n[0, j]
            J33n[-1, j] = J33n[0, j]
            g11n[-1, j] = g11n[0, j]
            g21n[-1, j] = g21n[0, j]
            g31n[-1, j] = g31n[0, j]
            g12n[-1, j] = g12n[0, j]
            g22n[-1, j] = g22n[0, j]
            g32n[-1, j] = g32n[0, j]
            g13n[-1, j] = g13n[0, j]
            g23n[-1, j] = g23n[0, j]
            g33n[-1, j] = g33n[0, j]

            invJn[i, -1] = invJn[i, 0]
            Jn[i, -1] = Jn[i, 0]
            J11n[i, -1] = J11n[i, 0]
            J21n[i, -1] = J21n[i, 0]
            J31n[i, -1] = J31n[i, 0]
            J12n[i, -1] = J12n[i, 0]
            J22n[i, -1] = J22n[i, 0]
            J32n[i, -1] = J32n[i, 0]
            J13n[i, -1] = J13n[i, 0]
            J23n[i, -1] = J23n[i, 0]
            J33n[i, -1] = J33n[i, 0]
            g11n[i, -1] = g11n[i, 0]
            g21n[i, -1] = g21n[i, 0]
            g31n[i, -1] = g31n[i, 0]
            g12n[i, -1] = g12n[i, 0]
            g22n[i, -1] = g22n[i, 0]
            g32n[i, -1] = g32n[i, 0]
            g13n[i, -1] = g13n[i, 0]
            g23n[i, -1] = g23n[i, 0]
            g33n[i, -1] = g33n[i, 0]

            invJn[-1, 0] = invJn[0, 0]
            Jn[-1, 0] = Jn[0, 0]
            J11n[-1, 0] = J11n[0, 0]
            J21n[-1, 0] = J21n[0, 0]
            J31n[-1, 0] = J31n[0, 0]
            J12n[-1, 0] = J12n[0, 0]
            J22n[-1, 0] = J22n[0, 0]
            J32n[-1, 0] = J32n[0, 0]
            J13n[-1, 0] = J13n[0, 0]
            J23n[-1, 0] = J23n[0, 0]
            J33n[-1, 0] = J33n[0, 0]
            g11n[-1, 0] = g11n[0, 0]
            g21n[-1, 0] = g21n[0, 0]
            g31n[-1, 0] = g31n[0, 0]
            g12n[-1, 0] = g12n[0, 0]
            g22n[-1, 0] = g22n[0, 0]
            g32n[-1, 0] = g32n[0, 0]
            g13n[-1, 0] = g13n[0, 0]
            g23n[-1, 0] = g23n[0, 0]
            g33n[-1, 0] = g33n[0, 0]

            invJn[0, -1] = invJn[0, 0]
            Jn[0, -1] = Jn[0, 0]
            J11n[0, -1] = J11n[0, 0]
            J21n[0, -1] = J21n[0, 0]
            J31n[0, -1] = J31n[0, 0]
            J12n[0, -1] = J12n[0, 0]
            J22n[0, -1] = J22n[0, 0]
            J32n[0, -1] = J32n[0, 0]
            J13n[0, -1] = J13n[0, 0]
            J23n[0, -1] = J23n[0, 0]
            J33n[0, -1] = J33n[0, 0]
            g11n[0, -1] = g11n[0, 0]
            g21n[0, -1] = g21n[0, 0]
            g31n[0, -1] = g31n[0, 0]
            g12n[0, -1] = g12n[0, 0]
            g22n[0, -1] = g22n[0, 0]
            g32n[0, -1] = g32n[0, 0]
            g13n[0, -1] = g13n[0, 0]
            g23n[0, -1] = g23n[0, 0]
            g33n[0, -1] = g33n[0, 0]

            invJn[-1, -1] = invJn[0, 0]
            Jn[-1, -1] = Jn[0, 0]
            J11n[-1, -1] = J11n[0, 0]
            J21n[-1, -1] = J21n[0, 0]
            J31n[-1, -1] = J31n[0, 0]
            J12n[-1, -1] = J12n[0, 0]
            J22n[-1, -1] = J22n[0, 0]
            J32n[-1, -1] = J32n[0, 0]
            J13n[-1, -1] = J13n[0, 0]
            J23n[-1, -1] = J23n[0, 0]
            J33n[-1, -1] = J33n[0, 0]
            g11n[-1, -1] = g11n[0, 0]
            g21n[-1, -1] = g21n[0, 0]
            g31n[-1, -1] = g31n[0, 0]
            g12n[-1, -1] = g12n[0, 0]
            g22n[-1, -1] = g22n[0, 0]
            g32n[-1, -1] = g32n[0, 0]
            g13n[-1, -1] = g13n[0, 0]
            g23n[-1, -1] = g23n[0, 0]
            g33n[-1, -1] = g33n[0, 0]
    return


def initiate_particles():
    global ns, npart, ppcx, ppcy, ppc, x, y, u, v, w, vth, udrift, vdrift, wdrift, vs, qs, ms, q, qm, m, J1, J2, J3, x0, y0
    if (npart == 0) | (GEM) | (restart):
        return
    for s in range(ns):
        xp0, yp0 = dx/(ppcx*2) + x0, dy/(ppcy*2) + y0
        xx, yy = np.mgrid[xp0:Lx-xp0:(nxc * ppcx * 1j), yp0:Ly-yp0:(nyc * ppcy * 1j)]
        x[s * (npart // ns):(s + 1) * (npart // ns)] = np.reshape(xx, npart // ns)
        y[s * (npart // ns):(s + 1) * (npart // ns)] = np.reshape(yy, npart // ns)
        if thermal:
            u[s * (npart // ns):(s + 1) * (npart // ns)], v[s * (npart // ns):(s + 1) * (npart // ns)], w[s * (npart // ns):(s + 1) * (npart // ns)] = (vth * vs[s] * 2 * (np.random.rand(npart // ns) - .5) for i in range(3))

        if drift:
            u[s * (npart // ns):(s + 1) * (npart // ns)], v[s * (npart // ns):(s + 1) * (npart // ns)], w[s * (npart // ns):(s + 1) * (npart // ns)] = (vth * vs[s] * 2 * (np.random.rand(npart // ns) - .5) for i in range(3))
            u[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * udrift
            v[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * vdrift
            w[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * wdrift


        if two_stream:
            u[s * (npart // ns):(s + 1) * (npart // ns)] = (vth * vs[s] * 2 * (np.random.rand(npart // ns) - .5))
            u[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * udrift
            v[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * vdrift
            w[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * wdrift
            u[s * (npart // ns):(s + 1) * (npart // ns):4] -= vs[s] * 2 * udrift
            v[s * (npart // ns):(s + 1) * (npart // ns):4] -= vs[s] * 2 * vdrift
            w[s * (npart // ns):(s + 1) * (npart // ns):4] -= vs[s] * 2 * wdrift
            u[s * (npart // ns)+1:(s + 1) * (npart // ns)+1:4] -= vs[s] * 2 * udrift
            v[s * (npart // ns)+1:(s + 1) * (npart // ns)+1:4] -= vs[s] * 2 * vdrift
            w[s * (npart // ns)+1:(s + 1) * (npart // ns)+1:4] -= vs[s] * 2 * wdrift

        if weibel:
            v[s * (npart // ns):(s + 1) * (npart // ns)] = (vth * vs[s] * 2 * (np.random.rand(npart // ns) - .5))
            u[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * udrift
            v[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * vdrift
            w[s * (npart // ns):(s + 1) * (npart // ns)] += vs[s] * wdrift
            u[s * (npart // ns):(s + 1) * (npart // ns):4] -= vs[s] * 2 * udrift
            v[s * (npart // ns):(s + 1) * (npart // ns):4] -= vs[s] * 2 * vdrift
            w[s * (npart // ns):(s + 1) * (npart // ns):4] -= vs[s] * 2 * wdrift
            u[s * (npart // ns)+1:(s + 1) * (npart // ns)+1:4] -= vs[s] * 2 * udrift
            v[s * (npart // ns)+1:(s + 1) * (npart // ns)+1:4] -= vs[s] * 2 * vdrift
            w[s * (npart // ns)+1:(s + 1) * (npart // ns)+1:4] -= vs[s] * 2 * wdrift

        q[0 + s * (npart // ns):npart // ns + s * (npart // ns)] = qs[s] * Lx * Ly / ((npart/ns) * 4 * pi)
        qm[0+s*(npart // ns):npart // ns + s * (npart // ns)] = qs[s] / ms[s]
        m[0 + s * (npart // ns):npart // ns + s * (npart // ns)] = q[0+s*(npart // ns):npart // ns + s * (npart // ns)] / qm[0+s*(npart // ns):npart // ns + s * (npart // ns)]

    Jx, Jy, Jz = particle_to_grid_J(x, y, u, v, w, q)
    J1, J2, J3 = cartesian_to_general_fieldC(Jx, Jy, Jz)
    return

def average_N2C(A):
    return .25 * (A[0:-1, 0:-1] + A[0:-1, 1:] + A[1:, 0:-1] + A[1:, 1:])


def energy(t):
    global E1, E2, E3, B1, B2, B3, J1, J2, J3, u, v, w, Utot, Ub, Ue, UJdotE, Ufield, Ufc, Uk, divB, charge, mom, cfl, EdotCurlBAv, BdotCurlEAv, sumDotCurlAv, deltaU, E_test, B_test
    UE1 = np.sum(.5 * Jc * dx * dy * (g11c * E1 * E1 + g12c * E1 * E2 + g13c * E1 * E3))/(4*pi)
    UE2 = np.sum(.5 * Jc * dx * dy * (g21c * E2 * E1 + g22c * E2 * E2 + g23c * E2 * E3))/(4*pi)
    UE3 = np.sum(.5 * Jc * dx * dy * (g31c * E3 * E1 + g32c * E3 * E2 + g33c * E3 * E3))/(4*pi)
    Ue[t] = UE1 + UE2 + UE3
    UB1 = np.sum(.5 * Jn[0:-1, 0:-1] * dx * dy * (g11n[0:-1, 0:-1] * B1[0:-1, 0:-1] * B1[0:-1, 0:-1] + g12n[0:-1, 0:-1] * B1[0:-1, 0:-1] * B2[0:-1, 0:-1] + g13n[0:-1, 0:-1] * B1[0:-1, 0:-1] * B3[0:-1, 0:-1]))/(4*pi)
    UB2 = np.sum(.5 * Jn[0:-1, 0:-1] * dx * dy * (g21n[0:-1, 0:-1] * B2[0:-1, 0:-1] * B1[0:-1, 0:-1] + g22n[0:-1, 0:-1] * B2[0:-1, 0:-1] * B2[0:-1, 0:-1] + g23n[0:-1, 0:-1] * B2[0:-1, 0:-1] * B3[0:-1, 0:-1]))/(4*pi)
    UB3 = np.sum(.5 * Jn[0:-1, 0:-1] * dx * dy * (g31n[0:-1, 0:-1] * B3[0:-1, 0:-1] * B1[0:-1, 0:-1] + g32n[0:-1, 0:-1] * B3[0:-1, 0:-1] * B2[0:-1, 0:-1] + g33n[0:-1, 0:-1] * B3[0:-1, 0:-1] * B3[0:-1, 0:-1]))/(4*pi)
    Ub[t] = UB1 + UB2 + UB3
    Ufield[t] = np.sum(Ub[t] + Ue[t])
    E_test[t] = UE1
    B_test[t] = Ub[t]

    Ex, Ey, Ez = general_to_cartesian_fieldC(E1, E2, E3)
    Bx, By, Bz = general_to_cartesian_fieldN(B1, B2, B3)
    UEc = np.sum(.5 * Jc * dx * dy * (Ex**2 + Ey**2 + Ez**2))/(4*pi)
    UBc = np.sum(.5 * Jn[0:-1, 0:-1] * dx * dy * (Bx[0:-1, 0:-1]**2 + By[0:-1, 0:-1]**2 + Bz[0:-1, 0:-1]**2))/(4*pi)

    Ufc[t] = UEc + UBc
    Uk[t] = np.sum(m * .5 * (u**2 + v**2 + w**2))
    Utot[t] = Ufield[t] + Uk[t]

    if npart > 0:
        cfl[t] = maxv = np.max(np.sqrt(u**2 + v**2 + w**2))*dt/np.minimum(dxmin, dymin)

    B1bar = (B1 + B1old) * .5
    B2bar = (B2 + B2old) * .5
    B3bar = (B3 + B3old) * .5
    E1bar = (E1 + E1old) * .5
    E2bar = (E2 + E2old) * .5
    E3bar = (E3 + E3old) * .5

    EdotCurlBAv[t] = np.sum(((g11c * E1bar * curlB(B1bar, B2bar, B3bar)[0] + g21c * E2bar * curlB(B1bar, B2bar, B3bar)[0] + g31c * E3bar * curlB(B1bar, B2bar, B3bar)[0]) +
                             (g12c * E1bar * curlB(B1bar, B2bar, B3bar)[1] + g22c * E2bar * curlB(B1bar, B2bar, B3bar)[1] + g32c * E3bar * curlB(B1bar, B2bar, B3bar)[1]) +
                             (g13c * E1bar * curlB(B1bar, B2bar, B3bar)[2] + g23c * E2bar * curlB(B1bar, B2bar, B3bar)[2] + g33c * E3bar * curlB(B1bar, B2bar, B3bar)[2])) * Jc)

    BdotCurlEAv[t] = np.sum(((g11n * -B1bar * curlE(E1bar, E2bar, E3bar)[0] + g21n * -B2bar * curlE(E1bar, E2bar, E3bar)[0] + g31n * -B3bar * curlE(E1bar, E2bar, E3bar)[0]) +
                             (g12n * -B1bar * curlE(E1bar, E2bar, E3bar)[1] + g22n * -B2bar * curlE(E1bar, E2bar, E3bar)[1] + g32n * -B3bar * curlE(E1bar, E2bar, E3bar)[1]) +
                             (g13n * -B1bar * curlE(E1bar, E2bar, E3bar)[2] + g23n * -B2bar * curlE(E1bar, E2bar, E3bar)[2] + g33n * -B3bar * curlE(E1bar, E2bar, E3bar)[2]))[0:-1, 0:-1] * Jn[0:-1, 0:-1])

    sumDotCurlAv[t] = EdotCurlBAv[t] + BdotCurlEAv[t]

    deltaUB1 = (g11n[0:-1, 0:-1] * B1[0:-1, 0:-1] * B1[0:-1, 0:-1] + g12n[0:-1, 0:-1] * B1[0:-1, 0:-1] * B2[0:-1, 0:-1] + g13n[0:-1, 0:-1] * B1[0:-1, 0:-1] * B3[0:-1, 0:-1]) - (g11n[0:-1, 0:-1] * B1old[0:-1, 0:-1] * B1old[0:-1, 0:-1] + g12n[0:-1, 0:-1] * B1old[0:-1, 0:-1] * B2old[0:-1, 0:-1] + g13n[0:-1, 0:-1] * B1old[0:-1, 0:-1] * B3old[0:-1, 0:-1])
    deltaUB2 = (g21n[0:-1, 0:-1] * B2[0:-1, 0:-1] * B1[0:-1, 0:-1] + g22n[0:-1, 0:-1] * B2[0:-1, 0:-1] * B2[0:-1, 0:-1] + g23n[0:-1, 0:-1] * B2[0:-1, 0:-1] * B3[0:-1, 0:-1]) - (g21n[0:-1, 0:-1] * B2old[0:-1, 0:-1] * B1old[0:-1, 0:-1] + g22n[0:-1, 0:-1] * B2old[0:-1, 0:-1] * B2old[0:-1, 0:-1] + g23n[0:-1, 0:-1] * B2old[0:-1, 0:-1] * B3old[0:-1, 0:-1])
    deltaUB3 = (g31n[0:-1, 0:-1] * B3[0:-1, 0:-1] * B1[0:-1, 0:-1] + g32n[0:-1, 0:-1] * B3[0:-1, 0:-1] * B2[0:-1, 0:-1] + g33n[0:-1, 0:-1] * B3[0:-1, 0:-1] * B3[0:-1, 0:-1]) - (g31n[0:-1, 0:-1] * B3old[0:-1, 0:-1] * B1old[0:-1, 0:-1] + g32n[0:-1, 0:-1] * B3old[0:-1, 0:-1] * B2old[0:-1, 0:-1] + g33n[0:-1, 0:-1] * B3old[0:-1, 0:-1] * B3old[0:-1, 0:-1])

    deltaUE1 = (g11c * E1 * E1 + g12c * E1 * E2 + g13c * E1 * E3) - (g11c * E1old * E1old + g12c * E1old * E2old + g13c * E1old * E3old)
    deltaUE2 = (g21c * E2 * E1 + g22c * E2 * E2 + g23c * E2 * E3) - (g21c * E2old * E1old + g22c * E2old * E2old + g23c * E2old * E3old)
    deltaUE3 = (g31c * E3 * E1 + g32c * E3 * E2 + g33c * E3 * E3) - (g31c * E3old * E1old + g32c * E3old * E2old + g33c * E3old * E3old)


    deltaU[t] = .5 * np.sum(deltaUB1 + deltaUB2 + deltaUB3 + deltaUE1 + deltaUE2 + deltaUE3)

    return

def myplot_field(A, title):
    plt.figure(title)
    plt.imshow(A.T, origin='lower')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\eta$')
    plt.colorbar()
    plt.title(title)i

def myplot_field_save(A, name, title):
    plt.figure(title)
    plt.imshow(A.T, origin='lower')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\eta$')
    plt.colorbar()
    plt.title(title)
    plt.savefig('figs\\' + name)
    plt.close(title)

def myplot_diagnostic(a, title, ylabel):
    plt.figure(title)
    plt.plot(a)
    plt.xlabel('time step')
    plt.ylabel(ylabel)
    plt.title(title)

def myplot_log_diagnostic(a, title, ylabel):
    plt.figure(title)
    plt.semilogy(a)
    plt.xlabel('time step')
    plt.ylabel(ylabel)
    plt.title(title)

def myplot_particles(x, y, u, v, q, title):
    plt.figure(title)
    plt.scatter(x, y, c=q, cmap='bwr')
    plt.quiver(x, y, u, v)
    plt.xlim(0, Lx)
    plt.ylim(0, Ly)

def myplot_stream_save(x, y, A1, A2, title, name):
    plt.figure(title)
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    plt.streamplot(y, x, A2, A1, transform= rot + base)
    plt.title(title)
    plt.savefig('figs\\' + name)
    plt.close(title)

def myplot_pert_map(xgrid, ygrid, field, xlabel='x', ylabel='y', title='title'):
    plt.figure(title + '_pert')
    plt.gca().set_aspect('equal')
    plt.pcolormesh(xgrid, ygrid, field, shading='auto')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def myplot_pert_map_save(xgrid, ygrid, field, name, title, xlabel='x', ylabel='y'):
    plt.figure(title)
    plt.gca().set_aspect('equal')
    plt.pcolormesh(xgrid, ygrid, field, shading='auto')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('figs\\' + name)
    plt.close(title)

def myplot_phase_space(x, u, title):
    fig = plt.figure(title)
    plt.scatter(x, u)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(title)

def myplot_phase_save(x, u, it):
    name = 'phase' + str(it)
    plt.figure(name)
    plt.scatter(x, u, color='blue')
    plt.xlim(0, 32)
    plt.ylim(-0.5, 0.5)
    plt.savefig('figs\\' + name + '.jpg')
    plt.close(name)


def save_restart_files(it, restart_path):
    np.save(restart_path + '/B1_' + str(it).zfill(5) + '.npy', B1)
    np.save(restart_path + '/B2_' + str(it).zfill(5) + '.npy', B2)
    np.save(restart_path + '/B3_' + str(it).zfill(5) + '.npy', B3)
    np.save(restart_path + '/E1_' + str(it).zfill(5) + '.npy', E1)
    np.save(restart_path + '/E2_' + str(it).zfill(5) + '.npy', E2)
    np.save(restart_path + '/E3_' + str(it).zfill(5) + '.npy', E3)
    np.save(restart_path + '/x_' + str(it).zfill(5) + '.npy', x)
    np.save(restart_path + '/y_' + str(it).zfill(5) + '.npy', y)
    np.save(restart_path + '/u_' + str(it).zfill(5) + '.npy', u)
    np.save(restart_path + '/v_' + str(it).zfill(5) + '.npy', v)
    np.save(restart_path + '/w_' + str(it).zfill(5) + '.npy', w)
    np.save(restart_path + '/q_' + str(it).zfill(5) + '.npy', q)
    np.save(restart_path + '/m_' + str(it).zfill(5) + '.npy', m)
    np.save(restart_path + '/qm_' + str(it).zfill(5) + '.npy', qm)
    return


def save_data_files(it, save_path):
    np.save(save_path + '/B1_' + str(it).zfill(5) + '.npy', B1)
    np.save(save_path + '/B2_' + str(it).zfill(5) + '.npy', B2)
    np.save(save_path + '/B3_' + str(it).zfill(5) + '.npy', B3)
    np.save(save_path + '/E1_' + str(it).zfill(5) + '.npy', E1)
    np.save(save_path + '/E2_' + str(it).zfill(5) + '.npy', E2)
    np.save(save_path + '/E3_' + str(it).zfill(5) + '.npy', E3)
    np.save(save_path + '/J1_' + str(it).zfill(5) + '.npy', J1)
    np.save(save_path + '/J2_' + str(it).zfill(5) + '.npy', J2)
    np.save(save_path + '/J3_' + str(it).zfill(5) + '.npy', J3)
    np.save(save_path + '/Utot_' + str(it).zfill(5) + '.npy', Utot[0:it+1])
    np.save(save_path + '/Ue_' + str(it).zfill(5) + '.npy', Ue[0:it+1])
    np.save(save_path + '/Ub_' + str(it).zfill(5) + '.npy', Ub[0:it+1])
    np.save(save_path + '/Uk_' + str(it).zfill(5) + '.npy', Uk[0:it+1])
    np.save(save_path + '/cfl_' + str(it).zfill(5) + '.npy', cfl[0:it+1])


def save_logfile():
    a = ''
    b = ''
    R = ''
    if sinus: b = '_sin'
    elif squared: b = '_square'
    elif skewed: b = '_skewed'
    elif CnC: b = '_CnC'
    elif hypertan: b = '_tanh'
    elif double_hypertan: b  ='_double_tanh'
    if weibel: a = '_weibel'
    elif two_stream: a = '_twostream'
    elif GEM: a = '_GEM'
    elif thermal: a = '_thermal'
    if restart: R = '_RESTART'
    filename1 = 'save_data\\logs\\log' + a + b + '_' + str(nx) + '_' + str(ny) + R + '.txt'
    filename2 = save_path + '/log' + a + b + '_' + str(nx) + '_' + str(ny) + R + '.txt'
    filename3 = restart_path + '/log' + a + b + '_' + str(nx) + '_' + str(ny) + R + '.txt'
    filenames = [filename1, filename2, filename3]
    for filename in filenames:
        with open(filename, 'w') as f:
            f.write('timesteps: ' +  str(nt) + '\n')
            f.write('dt: ' + str(dt)+ '\n')
            f.write('dx: ' + str(dx) + '\n')
            f.write('u_drift: ' + str(udrift) + '\n')
            f.write('v_drift: ' + str(vdrift) + '\n')
            f.write('w_drift: ' + str(wdrift) + '\n')
            f.write('v_thermal: ' + str(vth) + '\n')
            f.write('number of particles: ' + str(npart) + '\n')
            f.write('particles per cell: ' + str(ppc) + '\n')
            f.write('number of species: ' + str(ns) + '\n')
            if double_smooth_hs:
                f.write('grid density ratio: ' + str(r) + '\n')
                f.write('width of high density regions: ' + str(h) + '\n')
            else:
                f.write('epsilon: ' + str(eps) + '\n')
            f.write('gridsize: ' + str(Lx) + 'x' + str(Ly) + '\n')
            f.write('gridcells: ' + str(nx) + 'x' + str(ny)+ '\n')
            f.write('smallest grid size x: ' + str(np.min(xn[1:, :] - xn[0:-1, :])) + '\n')
            f.write('largest grid size x: ' + str(np.max(xn[1:, :] - xn[0:-1, :])) + '\n')
            f.write('smallest grid size y: ' + str(np.min(yn[:, 1:] - yn[:, 0:-1])) + '\n')
            f.write('largest grid size y: ' + str(np.max(yn[:, 1:] - yn[:, 0:-1])) + '\n')
            f.write('data save location: ' + save_path + '\n')
            if restart:
                f.write('RESTART\n')
                f.write('Restarted from file: ' + restart_location + '\n')
                f.write('Restarted from timestep: ' + str(restart_time + 1) + '\n')


def create_file_structure():
    global save_path, restart_path
    d = datetime.datetime.now()
    if double_hypertan:
        a = 'Dtanh'
        e = str(eps).replace('.', '')
    elif double_smooth_hs:
        a = 'DSHS'
        e = str(r) + '_' + str(h) + '_' + str(s)
    elif sinus:
        a = 'sin'
        e = str(eps).replace('.', '')
    elif hypertan:
        a = 'tanh'
        e = str(eps).replace('.', '')
    else:
        a = 'cart'
        e = str(eps).replace('.', '')
    b = ''
    if GEM:
        b = 'GEM'
    elif two_stream:
        b = '2stream'
    elif weibel:
        b = 'weibel'
    if restart:
        b = 'RESTART'

    casestamp = a + '_' + b + '_' + str(nx) + '_' + str(ny) + '_' + str(ppc) + '_' + e + '_' + str(nt) + '_' + str(dt).replace('.', '')
    timestamp = "_%04d_%02d_%02d_%02d_%02d" % (d.year, d.month, d.day, d.hour, d.minute)
    save_path = 'save_data/data/' + casestamp + timestamp
    print(save_path)
    os.mkdir(save_path)
    restart_path = 'save_data/restart/' + casestamp + timestamp
    os.mkdir(restart_path)


def main():
    global time_init, time_energy, time_IO, save_path, restart_path, B1, B2, B3, x, y, u, v, w, q
    print('Start')
    start = time.time()
    print('Initialising geometry')
    initiate_geometry()
    print('Initialising particles')
    initiate_particles()
    print('create files')
    if save_data:
        create_file_structure()
        save_logfile()
    print('recompiling numba functions')
    recompile_nb_code()
    time_init += time.time() - start
    print('Main loop')
    for it in range(0, nt):
        print(it)
        sol = newton_krylov_iterator(NK_solver_flag)
        maxwell_solver(sol)
        start = time.time()
        energy(it)
        time_energy += time.time() - start
        start = time.time()
        if (((((it+1) % frequency_image) == 0) or (it == nt-1)) & images):
            #name = 'streamlines_' + str(it+1)
            #myplot_stream_save(xi_n, eta_n, B1, B2, name, name + '.png')
            myplot_phase_save(x, u, it+1)
            #name = 'B2_' + str(it+1)
            #myplot_field_save(B2, name, name)
            name = 'B1_' + str(it+1)
            myplot_field_save(B1, name, name)
            #name = 'J3_' + str(it+1)
            #myplot_field_save(J3, name, name)
            #name = 'B2pert_' + str(it+1)
            #myplot_pert_map_save(xn, yn, B2, name, name)
            #name = 'B1pert_' + str(it+1)
            #myplot_pert_map_save(xn, yn, B1, name, name)
            #name = 'J3pert_' + str(it+1)
            #myplot_pert_map_save(xn, yn, J3, name, name)
            #val = np.sqrt(J1**2 + J2**2 + J3**2)
        if (((((it+1) % frequency_save) == 0) or (it == nt-1)) & save_data):
            save_data_files(it+1, save_path)
        if (((((it+1) % frequency_restart) == 0) or (it == nt-1)) & save_restart_data):
            save_restart_files(it+1, restart_path)
        time_IO += time.time() - start



start = time.time()
main()
time_tot += time.time() - start


myplot_field(B1, 'B1')
myplot_field(B2, 'B2')
myplot_field(B3, 'B3')
myplot_field(E1, 'E1')
myplot_field(E2, 'E2')
myplot_field(J1, 'J1')
myplot_field(J2, 'J2')
myplot_field(J3, 'J3')
myplot_particles(x, y, u, v, q, 'particles')
myplot_diagnostic(Ufield, 'energy field', 'energy')
#myplot_diagnostic(Ufc, 'energy field alt', 'energy')
#myplot_diagnostic(Ufield - Ufc, 'difference between Ufield and Ufc', 'energy')
myplot_log_diagnostic(np.abs((Ufield-Ufield[0])/Ufield[0]), 'normalised field energy', 'energy')
myplot_diagnostic(Uk, 'energy particles', 'energy')
myplot_diagnostic(Utot, 'total energy', 'energy')
myplot_log_diagnostic(np.abs((Utot-Utot[0])/Utot[0]), 'normalised total energy', 'energy')
#myplot_diagnostic(Ex2, 'ex2', 'ex2')
#myplot_diagnostic(Exi2, 'exi2', 'exi2')
#myplot_diagnostic(Exi2-Ex2, 'should be zero', 'is it zero?')
#myplot_diagnostic(UJdotE, 'energy from JdotE', 'energy')
myplot_diagnostic(cfl, 'CFL max', 'CFL number')
#myplot_diagnostic(EdotCurlB, 'EdotCurlB', 'EdotCurlB')
#myplot_diagnostic(BdotCurlE, 'BdotCurlE', 'BdotCurlE')
#myplot_diagnostic(sumDotCurl, 'sumDotCurl', 'sumDotCurl')
myplot_diagnostic(sumDotCurlAv, 'sumDotCurlAv', 'sumDotCurlAv')
#myplot_diagnostic(deltaU, 'deltaU', 'deltaU')
myplot_phase_space(x, u, 'phase space')
myplot_pert_map(xn, yn, B3, xlabel='x', ylabel='y', title='B3')

#myplot_diagnostic(Uratio, 'Energy ratio fields to particles')
#myplot_diagnostic(Ukm, 'Kinetic energy from moments')
#myplot_diagnostic(Ufield + Ukm, 'Total energy from moments')

#UfieldNormal = (Ufield-np.min(Ufield)) / np.max(Ufield-np.min(Ufield))
#UkNormal = (Uk-np.min(Uk)) / np.max(Uk-np.min(Uk))
#UtotalNormal = UfieldNormal + UkNormal
#
#myplot_diagnostic(UfieldNormal, 'energy fields normalised')
#myplot_diagnostic(UkNormal, 'energy particles normalised')
#myplot_diagnostic(UtotalNormal, 'total energy from normalised kinetic and field energies')


print('')
print('###########################--PARAMETERS--###########################################################')
print('timesteps: ', nt)
print('dt: ', dt)
print('number of particles: ', npart)
print('gridsize: ', nx, 'x', ny)

print('')
print('#############################--TIMING--#############################################################')
print('Initiation: ', time_init)
print('Mover: ', time_mover)
print('Newton: ', time_newt)
print('Maxwell: ', time_maxwell)
print('Grid to particles: ', time_f2p)
print('Particles to grid: ', time_p2f)
print('Cartesian to general particles: ', time_c2g_part)
print('Cartesian to general fields: ', time_c2g_field)
print('General to cartesian fields: ', time_g2c_field)
print('Energy: ', time_energy)
print('Linear algebra: ', time_linalg)
print('Physics to krylov: ', time_phys2kry)
print('Krylov to physics: ', time_kry2phys)
print('I/O: ', time_IO)
print('Total: ', time_tot)
print('Sum: ', time_init + time_mover + time_newt + time_maxwell + time_f2p + time_p2f + time_c2g_part + time_c2g_field + time_g2c_field + time_phys2kry + time_kry2phys + time_energy + time_linalg)


plt.show()
