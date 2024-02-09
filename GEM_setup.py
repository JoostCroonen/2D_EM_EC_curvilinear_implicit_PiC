import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numba import njit
from numpy import tanh, sqrt, sin, cos, cosh

double_smooth_hs = True
perturb = False

# Simulation parameters
nx, ny = 32, 64
nxc, nyc = nx, ny
nxn, nyn = nxc+1, nyc+1
nxp, nyp = 32, 64
nxcp, nycp = nxp, nyp
nxnp, nynp = nxcp+1, nycp+1
Lx, Ly = 16, 32
x0, y0 = 0, 0                       # offset of the origin
dx, dy = Lx / nxc, Ly / nyc
invdx, invdy = 1 / dx, 1 / dy
dt = .1*dx                            # time step size
nt = 100                           # number of time steps
ns = 4                             # number of species in plasma
ppcx = 4                           # particles per cell in the x direction
ppcy = 4
ppc = ppcx * ppcy
npart = ns * ppcx * ppcy * nxcp * nycp

np.random.seed(42)

eps = 0.0       # perturbation parameter
r = 10          # density ratio
h = 8           # high denisty region width
s = 5           # sharpness parameter of the hypertan heaviside approximation
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


# Grid
xi_c, eta_c = np.mgrid[x0 + dx / 2:x0 + Lx - dx / 2:(nxc * 1j), y0 + dy / 2:y0 + Ly - dy / 2:(nyc * 1j)]
xi_n, eta_n = np.mgrid[x0 + 0:x0 + Lx:(nxn * 1j), y0 + 0:y0 + Ly:(nyn * 1j)]
xc = xi_c.copy()
yc = eta_c.copy()
xn = xi_n.copy()
yn = eta_n.copy()
if double_smooth_hs:
    a = 'double_smooth_hs_'
    e = str(r) + '_' + str(h) + '_' + str(s)
    name = 'save_data\\grids\\' + a + str(nx) + '_' + str(ny) + '_' + str(Lx) + '_' + str(Ly) + '_' + e
    xc = np.load(name + '_xc.npy')
    yc = np.load(name + '_yc.npy')
    xn = np.load(name + '_xn.npy')
    yn = np.load(name + '_yn.npy')


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

B0 = 0.1
Psi0 = 0.8
n0 = 0.5/(4*np.pi) # number density of each species
ncsn0 = 3.0 # Density ratio at the center of the CS
delta = 0.5
vth_cs = B0/np.sqrt(16 * np.pi * ncsn0 * n0)
vb_cs = B0/(8 * np.pi * ncsn0 * n0 * delta)
print('vth_cs', vth_cs)
print('vb_cs', vb_cs)
B1 = B0 * (-np.tanh((yn - (Ly/4))/delta) - np.tanh((-yn + (3*Ly/4))/delta) + 1)
nd = B0 /(delta*8*np.pi*vb_cs) * ((np.cosh((-yc+Ly/4)/delta))**(-2) + (np.cosh((-yc+3*Ly/4)/delta))**(-2))

pertx1 = - Psi0 * B0 * np.cos(2*np.pi*xn/Lx) * np.sin(np.pi*(yn-Ly/4)/Ly) * np.exp(-(yn-Ly/4)**2/2) * np.exp(-(xn-Lx/2)**2/2)
perty1 = Psi0 * B0 * np.sin(2*np.pi*xn/Lx) * np.cos(np.pi*(yn-Ly/4)/Ly) * np.exp(-(yn-Ly/4)**2/2) * np.exp(-(xn-Lx/2)**2/2)
pertx2 = Psi0 * B0 * np.cos(2*np.pi*xn/Lx) * np.sin(np.pi*(yn-3*Ly/4)/Ly) * np.exp(-(yn-3*Ly/4)**2/2) * np.exp(-(xn-Lx/2)**2/2)
perty2 = - Psi0 * B0 * np.sin(2*np.pi*xn/Lx) * np.cos(np.pi*(yn-3*Ly/4)/Ly) * np.exp(-(yn-3*Ly/4)**2/2) * np.exp(-(xn-Lx/2)**2/2)

B1_pert = B1
B2_pert = B2

if perturb:
    B1_pert += pertx1 + pertx2
    B2_pert += perty1 + perty2

# Particles
vth = 0.01
vs = np.array([+1, -1, -1, +1])
qs = np.array([-1, +1, -1, +1])
wp = n0 * Lx*Ly/(npart // ns)
ms = np.array([1, 1, 1, 1])
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


def ddx_n2c(A):
    return ((A[1:, :-1] + A[1:, 1:]) * .5 - (A[:-1, :-1] + A[:-1, 1:]) * .5) * invdx

def ddy_n2c(A):
    return ((A[:-1, 1:] + A[1:, 1:]) * .5 - (A[:-1, :-1] + A[1:, :-1]) * .5) * invdy

def curlB(A1, A2, A3):
    curl1 = (ddy_n2c(A3))
    curl2 = (-ddx_n2c(A3))
    curl3 = (ddx_n2c(A2) - ddy_n2c(A1))
    return curl1, curl2, curl3

@njit("UniTuple(f8[:,:], 3)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])")
def particle_to_grid_J(xk, yk, uk, vk, wk, q):
    ''' Interpolation particle to grid - current J -> c
    '''

    Jx = np.zeros((nxc, nyc), np.float64)
    Jy = np.zeros((nxc, nyc), np.float64)
    Jz = np.zeros((nxc, nyc), np.float64)

    for i in range(xk.size):
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

        Jx[i1, j1] += wx1 * wy1 * q[i] * uk[i] * invdx * invdy
        Jy[i1, j1] += wx1 * wy1 * q[i] * vk[i] * invdx * invdy
        Jz[i1, j1] += wx1 * wy1 * q[i] * wk[i] * invdx * invdy
        Jx[i2, j1] += wx2 * wy1 * q[i] * uk[i] * invdx * invdy
        Jy[i2, j1] += wx2 * wy1 * q[i] * vk[i] * invdx * invdy
        Jz[i2, j1] += wx2 * wy1 * q[i] * wk[i] * invdx * invdy
        Jx[i1, j2] += wx1 * wy2 * q[i] * uk[i] * invdx * invdy
        Jy[i1, j2] += wx1 * wy2 * q[i] * vk[i] * invdx * invdy
        Jz[i1, j2] += wx1 * wy2 * q[i] * wk[i] * invdx * invdy
        Jx[i2, j2] += wx2 * wy2 * q[i] * uk[i] * invdx * invdy
        Jy[i2, j2] += wx2 * wy2 * q[i] * vk[i] * invdx * invdy
        Jz[i2, j2] += wx2 * wy2 * q[i] * wk[i] * invdx * invdy

    return Jx, Jy, Jz

for s in range(ns):
    xp0, yp0 = dx / (ppcx * 2) + x0, dy / (ppcy * 2) + y0
    xx, yy = np.mgrid[xp0:Lx - xp0:(nxcp * ppcx * 1j), yp0:Ly - yp0:(nycp * ppcy * 1j)]
    x[s * (npart // ns):(s + 1) * (npart // ns)] = np.reshape(xx, npart // ns)
    y[s * (npart // ns):(s + 1) * (npart // ns)] = np.reshape(yy, npart // ns)
    #m[s * (npart // ns):(s + 1) * (npart // ns)] = ms[s]

for s in range(0, 2):
    u[s * (npart // ns):(s + 1) * (npart // ns)] = vth*2*(np.random.rand(npart//ns)-.5) #vth * np.random.normal(npart // ns)
    v[s * (npart // ns):(s + 1) * (npart // ns)] = vth*2*(np.random.rand(npart//ns)-.5) #vth * np.random.normal(npart // ns)
    w[s * (npart // ns):(s + 1) * (npart // ns)] = vth*2*(np.random.rand(npart//ns)-.5) #vth * np.random.normal(npart // ns)
    q[s * (npart // ns):(s + 1) * (npart // ns)] = qs[s] * wp
    qm[s * (npart // ns):(s + 1) * (npart // ns)] = qs[s] / ms[s]
    m[s * (npart // ns):(s + 1) * (npart // ns)] = np.abs(q[s * (npart // ns):(s + 1) * (npart // ns)])

for s in range(2, 4):
    u[s * (npart // ns):(s + 1) * (npart // ns)] = vth_cs*2*(np.random.rand(npart//ns)-.5) #vth_cs * np.random.normal(npart // ns)
    v[s * (npart // ns):(s + 1) * (npart // ns)] = vth_cs*2*(np.random.rand(npart//ns)-.5) #vth_cs * np.random.normal(npart // ns)
    w[s * (npart // ns):(s + 1) * (npart // ns)] = vth_cs*2*(np.random.rand(npart//ns)-.5) + vs[s]*vb_cs #vth_cs * np.random.normal(npart // ns) + vs[s]*vb_cs
    q[s * (npart // ns):(s + 1) * (npart // ns)] = qs[s] * wp * ncsn0 * ((np.cosh((-y[s*(npart//ns):(s+1)*(npart//ns)]+Ly/4)/delta))**(-2) + (np.cosh((-y[s*(npart//ns):(s+1) * (npart//ns)]+3*Ly/4)/delta))**(-2))
    qm[s * (npart // ns):(s + 1) * (npart // ns)] = qs[s] / ms[s] #q[s * (npart // ns):(s + 1) * (npart // ns)] / ms[s]
    m[s * (npart // ns):(s + 1) * (npart // ns)] = np.abs(q[s * (npart // ns):(s + 1) * (npart // ns)])

idi = np.argwhere(y>Ly/2)
w[idi] = -w[idi]

print('culling')
idx = np.argwhere(np.abs(q)<1e-7)
print('wp',wp)
print('q')
print(q)
print('Size before',q.size)
q = np.delete(q, idx)
print('Size after',q.size)
qm = np.delete(qm, idx)
x = np.delete(x, idx)
y = np.delete(y, idx)
u = np.delete(u, idx)
v = np.delete(v, idx)
w = np.delete(w, idx)
m = np.delete(m, idx)
npart = np.array([q.size])
numspc3 = (npart[0] - 2 * nxc * nyc * ppcx * ppcy)//2
print(idx.size)
print(numspc3)
print(nxc * nyc * ppcx * ppcy)
print((nxc * nyc * ppcx * ppcy)*2 + numspc3 * 2)
print(npart[0])
print(ns*ppcx*ppcy*nxc*nyc)

def cartesian_to_general_part(x, y):
    '''To convert the particles position from Cartesian geom. to General geom.
    '''
    if double_smooth_hs:
        xi = x
        eta = L / mv * (y * (1 - (0.5 + 0.5 * tanh(s * (y - b1)))) + (r * y - o1) * (0.5 + 0.5 * tanh(s * (y - b1)))
                        - (r * y - o1) * (0.5 + 0.5 * tanh(s * (y - b2))) + (y + o2) * (0.5 + 0.5 * tanh(s * (y - b2)))
                        - (y + o2) * (0.5 + 0.5 * tanh(s * (y - b3))) + (r * y - o3) * (0.5 + 0.5 * tanh(s * (y - b3)))
                        - (r * y - o3) * (0.5 + 0.5 * tanh(s * (y - b4))) + (y + o4) * (0.5 + 0.5 * tanh(s * (y - b4))))
    else:
        xi = x
        eta = y
    return xi, eta

xi1, xi2 = cartesian_to_general_part(x, y)
Jx, Jy, Jz = particle_to_grid_J(xi1, xi2, u, v, w, q)
curlB1, curlB2, curlB3 = curlB(B1, B2, B3)

print('plotting')

plt.figure('B')
plt.quiver(xi_n, eta_n, B1, B2)

plt.figure('curlB3 - 4piJz')
plt.imshow((curlB(B1, B2, B3)[2] - 4*np.pi*Jz).T, origin='lower')
plt.colorbar()

plt.figure('curlB3 - 4piJz propper')
plt.gca().set_aspect('equal')
plt.pcolormesh(xn, yn, (curlB3 - 4*np.pi*Jz), shading='auto', edgecolor='k', linewidth=0.0, cmap='jet')
plt.colorbar()

plt.figure('curlB3')
plt.imshow((curlB(B1, B2, B3)[2]).T, origin='lower')
plt.colorbar()

plt.figure('B1')
plt.imshow(B1.T, origin='lower')

plt.figure('B1_propper')
plt.pcolormesh(xn, yn, B1, shading='auto', edgecolor='k', linewidth=0.5)

plt.figure('pert1')
plt.imshow(pertx1.T + pertx2.T, origin='lower')

plt.figure('pert2')
plt.imshow(perty1.T + perty2.T, origin='lower')

plt.figure('B_pert')
plt.quiver(xi_n, eta_n, B1 + pertx1 + pertx2, B2 + perty1 + perty2, scale=150)

#plt.figure('particles')
#plt.scatter(x, y, c=q)

plt.figure('Jz')
plt.imshow(Jz.T, origin='lower')

plt.figure('4piJz')
plt.imshow(4*np.pi*Jz.T, origin='lower')
plt.colorbar()

print('saving')

np.save('save_data/init/GEM_pert_x.npy', x)
np.save('save_data/init/GEM_pert_y.npy', y)
np.save('save_data/init/GEM_pert_u.npy', u)
np.save('save_data/init/GEM_pert_v.npy', v)
np.save('save_data/init/GEM_pert_w.npy', w)
np.save('save_data/init/GEM_pert_B1.npy', B1_pert)
np.save('save_data/init/GEM_pert_B2.npy', B2_pert)
np.save('save_data/init/GEM_pert_q.npy', q)
np.save('save_data/init/GEM_pert_qm.npy', qm)
np.save('save_data/init/GEM_pert_m.npy', m)
np.save('save_data/init/GEM_pert_npart.npy', npart)


B1 = np.load('save_data/init/GEM_pert_B1.npy')
B2 = np.load('save_data/init/GEM_pert_B2.npy')
B3 = np.zeros_like(B1)
x = np.load('save_data/init/GEM_pert_x.npy')
y = np.load('save_data/init/GEM_pert_y.npy')
u = np.load('save_data/init/GEM_pert_u.npy')
v = np.load('save_data/init/GEM_pert_v.npy')
w = np.load('save_data/init/GEM_pert_w.npy')
q = np.load('save_data/init/GEM_pert_q.npy')
curlB1, curlB2, curlB3 = curlB(B1, B2, B3)
xi1, xi2 = cartesian_to_general_part(x, y)
J1, j2, J3 = particle_to_grid_J(xi1, xi2, u, v, w, q)
plt.figure('init2')
plt.gca().set_aspect('equal')
plt.pcolormesh(xn, yn, (curlB3 - 4 * np.pi * J3), shading='auto', edgecolor='k', linewidth=0.0, cmap='jet')
plt.colorbar()
plt.show()
