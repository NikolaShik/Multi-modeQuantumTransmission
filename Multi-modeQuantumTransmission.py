import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize, fsolve
import random as rnd
from scipy.linalg import sqrtm, expm, sinm, cosm
from joblib import Parallel, delayed
import multiprocessing
import mpmath as mp
from scipy.sparse.linalg import eigsh
import os
from skfem import *
try:
	from skfem.helpers import dot, grad
except ImportError:
	from skfem.models.helpers import dot, grad

# Physical constants (SI)
hbar = 1.054571817e-34
e = 1.602176634e-19
m = 0.0665 * (9.1093837139e-31)
nm = 1e-9

# Number of modes
K = 10

# Strucutre parameters:
Vattr = -0.4   # Potential in insertion, eV
h = 3.5         # Insertion width, nm
H = 5.0         # Waveguide width, nm
V0 = 1.0        # Potential outside waveguide, eV
Delta = 5.0     # Distance to infinite walls, nm

# Number of transverse states calculated
nstates = K

scale = hbar**2 / (2 * m * e * nm**2)

print(f"Starting {K}-mode calculation with Vattr = {Vattr} eV.")

################# TRANSVERSE PROBLEM WITH CACHE (FINITE ELEMENT METHOD AND CACHE - THANKS TO DeepSeek) ###################
# ---------- mesh with breakpoints ----------
dx_max = 0.01 * H          # max element length (can be reduced further)
n1 = max(1, int(np.ceil(Delta / dx_max)))
n2 = max(1, int(np.ceil(h / dx_max)))
n3 = max(1, int(np.ceil((H - h) / dx_max)))
n4 = max(1, int(np.ceil(Delta / dx_max)))

x1 = np.linspace(-Delta, 0, n1 + 1)
x2 = np.linspace(0, h, n2 + 1)[1:]
x3 = np.linspace(h, H, n3 + 1)[1:]
x4 = np.linspace(H, H + Delta, n4 + 1)[1:]
nodes = np.concatenate([x1, x2, x3, x4])

mesh = MeshLine(nodes)

# ---------- higher‑order element (p = 4 here) ----------
p_order = 4                     # <-- change to 5,6,... for even more accuracy
element = ElementLinePp(p_order)
basis = Basis(mesh, element)

# ---------- forms ----------
@BilinearForm
def stiff(u, v, w):
	return dot(grad(u), grad(v))

@BilinearForm
def mass(u, v, w):
	return u * v

def make_potential_form(V_func):
	@BilinearForm
	def pot(u, v, w):
		x = w.x[0]
		return V_func(x) * u * v
	return pot

# Resonator transverse potential
def Vres_func(x):
	return np.where(x <= 0, V0,
					np.where(x <= h, Vattr,
							 np.where(x <= H, 0.0, V0)))

# Waveguide transverse potential
def Vwav_func(x):
	return np.where((x <= 0) | (x >= H), V0, 0.0)

# ---------- assembly ----------
S = asm(stiff, basis)
M = asm(mass, basis)
M_Vres = asm(make_potential_form(Vres_func), basis)
M_Vwav = asm(make_potential_form(Vwav_func), basis)

A_res = scale * S + M_Vres
A_wav = scale * S + M_Vwav

# Define functions that identify the left / right boundaries by coordinate.
def left(x):
	return np.isclose(x[0], -Delta)          # x == -Delta

def right(x):
	return np.isclose(x[0], H + Delta)       # x == H+Delta

left_dofs = basis.get_dofs(left)    # all DOFs on the left vertex
right_dofs = basis.get_dofs(right)  # all DOFs on the right vertex
boundary_dofs = np.hstack((left_dofs, right_dofs))

A_res_int, M_int = condense(A_res, M, D=boundary_dofs, expand=False)
A_wav_int, _      = condense(A_wav, M, D=boundary_dofs, expand=False)

# normalise vectors function
def normalize(Mmat, evecs):
	for i in range(evecs.shape[1]):
		v = evecs[:, i]
		nrm = np.sqrt(v @ Mmat @ v)
		evecs[:, i] /= nrm

# Check cache
cache_file = 'TransverseModesCache.npy'
params = (Vattr, h, H, V0, Delta, nstates)
cache = {}
if os.path.exists(cache_file):
	# allow_pickle=True is required because the file contains a dict
	cache = np.load(cache_file, allow_pickle=True).item()
if params in cache:
	WT, RT, mu = cache[params]
	print(f"Transverse problem was cached.")
else:
	# ---------- solve ----------
	print(f"Transverse problem was NOT cached. Calculating...")
	evals_res, evecs_res = eigsh(A_res_int, k=nstates, M=M_int, which='SA')
	evals_wav, evecs_wav = eigsh(A_wav_int, k=nstates, M=M_int, which='SA')

	WT = np.array([evals_wav]).T
	RT = np.array([evals_res]).T

	normalize(M_int, evecs_res)
	normalize(M_int, evecs_wav)

	mu = evecs_res.T @ M_int @ evecs_wav

	cache[params] = (WT, RT, mu)
	np.save(cache_file, cache, allow_pickle=True)


det_val = np.linalg.det(mu @ mu.T)

print(f"Eigenvalues Vres (gRes)\n{RT[0:nstates]}")
print(f"Eigenvalues Vwav (gWav)\n{WT[0:nstates]}")
np.set_printoptions(precision=8, linewidth=250, suppress=False)
print(f"Matrix mu\n{mu}")
print(f"Det[mu . Transpose[mu]] = {det_val:.8}")
###########################################

# mu-matrix restriction to K modes
mu_uK, mu_sK, mu_vhK = np.linalg.svd(mu[0:K, 0:K])
mu = (mu_uK @ mu_vhK)
print(f"Unitary approximation for matrix mu\n{mu}")
print(f"Det[mu . Transpose[mu]] = {np.linalg.det(mu @ mu.T):.8}")

# Resonator length in nm
L = 1

def k(ENERGY):
	return 1e-9 / hbar * np.sqrt(2 * m * e * (ENERGY - WT[0:K,0:1] + 0j), dtype=np.complex128)

def q(ENERGY):
	return 1e-9 / hbar * np.sqrt(2 * m * e * (ENERGY - RT[0:K,0:1] + 0j), dtype=np.complex128)

# Waveguide distance corresponding to Pi phase shift at ENERGY = U
DPi = np.abs(np.pi/k(WT[1,0])[0])

def q_matrix(ENERGY):
	return np.diag(np.transpose(q(ENERGY))[0])

def kk_matrix(ENERGY):
	return np.diag(np.transpose(k(ENERGY))[0])

def ki_matrix(ENERGY, i):
	if (i % 2) == 0:
		return kk_matrix(ENERGY)
	else:
		return q_matrix(ENERGY)

def expk(ENERGY, i, x, x0):
	kk = ki_matrix(ENERGY, i)
	if i == 0:
		return expm(1j * kk * (x - x0[0]))
	else:
		return expm(1j * kk * (x - x0[i - 1]))

def mui(i):
	if (i % 2) == 0:
		return mu
	else:
		return mu.T

def S_matrix(ENERGY, i, x0):
	# Create 2Kx2K matrix by blocks
	S = np.zeros((2 * K, 2 * K), dtype=np.complex128)

	# Precalculation
	ki = ki_matrix(ENERGY, i)
	kip1 = ki_matrix(ENERGY, i + 1)
	expki = expk(ENERGY, i, x0[i], x0)
	Di = np.linalg.inv(ki @ mui(i).T + mui(i).T @ kip1)
	tildeDi = np.linalg.inv(mui(i) @ ki + kip1 @ mui(i))
	Fi = mui(i).T @ kip1 - ki @ mui(i).T
	tildeFi = mui(i) @ ki - kip1 @ mui(i)

	# Top-left block
	S[0:K, 0:K] = 2 * (Di @ ki @ expki)
	
	# Top-right block
	S[0:K, K:(2 * K)] = Di @ Fi
	
	# Bottom-left block
	S[K:(2 * K), 0:K] = (expki @ tildeDi @ tildeFi @ expki)
	
	# Bottom-right block
	S[K:(2 * K), K:(2 * K)] = 2 * (expki @ tildeDi  @ kip1)

	return S

def Transmission_S(ENERGY, LL, D1, D2, D3, D4):
	# Create 2Kx2K matrix by blocks
	S = np.zeros((2 * K, 2 * K), dtype=np.complex128)
	Stemp = S

	# x0
	x0 = np.array([
	0,
	LL,
	LL + D1,
	2 * LL + D1,
	2 * LL + D1 + D2,
	3 * LL + D1 + D2,
	3 * LL + D1 + D2 + D3,
	4 * LL + D1 + D2 + D3,
	4 * LL + D1 + D2 + D3 + D4,
	5 * LL + D1 + D2 + D3 + D4
	])

	S = S_matrix(ENERGY, 0, x0)
	
	for i in np.linspace(1, x0.shape[0] - 1, x0.shape[0] - 1, dtype = 'int'):
		Stemp = np.zeros((2 * K, 2 * K), dtype=np.complex128)
		Sip1 = S_matrix(ENERGY, i, x0)
		D = np.linalg.inv(np.eye(K) - Sip1[K:(2 * K), 0:K] @ S[0:K, K:(2 * K)])

		# Top-left block
		Stemp[0:K, 0:K] = Sip1[0:K, 0:K] @ (np.eye(K) + S[0:K, K:(2 * K)] @ D @ Sip1[K:(2 * K), 0:K]) @ S[0:K, 0:K]

		# Top-right block
		Stemp[0:K, K:(2 * K)] = Sip1[0:K, K:(2 * K)] + Sip1[0:K, 0:K] @ S[0:K, K:(2 * K)] @ D @ Sip1[K:(2 * K), K:(2 * K)]

		# Bottom-left block
		Stemp[K:(2 * K), 0:K] = S[K:(2 * K), 0:K] + S[K:(2 * K), K:(2 * K)] @ D @ Sip1[K:(2 * K), 0:K] @ S[0:K, 0:K]

		# Bottom-right block
		Stemp[K:(2 * K), K:(2 * K)] = S[K:(2 * K), K:(2 * K)] @ D @ Sip1[K:(2 * K), K:(2 * K)]

		S = Stemp

	return np.abs(S[0, 0])**2

def plot_transmission_density(energy_range, param_range, 
							 num_energy_points=500, num_param_points=50,
							 log_scale=False, n_jobs=-1, backend='loky'):
	
	# Create meshgrid for ENERGY and D4
	energy = np.linspace(energy_range[0], energy_range[1], num_energy_points)
	param = np.linspace(param_range[0], param_range[1], num_param_points)
	ENERGY_mesh, PARAM_mesh = np.meshgrid(energy, param)
	
	# Flatten the mesh for parallel processing
	energy_flat = ENERGY_mesh.flatten()
	param_flat = PARAM_mesh.flatten()
	
	# Calculate number of CPU cores
	num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
	print(f"Using {num_cores} CPU cores for parallel computation...")
	
	# Parallel computation of transmission values
	transmission_flat = Parallel(n_jobs=n_jobs, backend=backend)(
		# Parameters of Transmission_S function: ENERGY, LL, D1, D2, D3, D4
		delayed(Transmission_S)(e, p, 3, 1, 1.6, 19) 
		for e, p in zip(energy_flat, param_flat)
	)
	
	# Reshape back to 2D
	transmission = np.array(transmission_flat).reshape(ENERGY_mesh.shape)
	
	# Create the plot
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Choose normalization
	if log_scale:
		# Handle potential zeros or negative values for log scale
		transmission_plot = np.maximum(transmission, 1e-10)  # Avoid log(0)
		norm = LogNorm()
		im = ax.pcolormesh(ENERGY_mesh, PARAM_mesh, transmission_plot, 
						  norm=norm, cmap='viridis', shading='auto')
	else:
		im = ax.pcolormesh(ENERGY_mesh, PARAM_mesh, transmission, 
						  cmap='viridis', shading='auto')
	
	# Add colorbar
	cbar = plt.colorbar(im, ax=ax)
	cbar.set_label('Transmission', fontsize=12)
	
	# Labels and title
	ax.set_xlabel('Energy', fontsize=12)
	ax.set_ylabel('Parameter', fontsize=12)
	ax.set_title(f'Transmission Density Plot (Parallel Computing)\n'
				f'Grid: {num_energy_points}×{num_param_points} = {num_energy_points * num_param_points} points | '
				f'Cores: {num_cores}', 
				fontsize=12)
	
	plt.tight_layout()
	plt.show()
	
	# Optional: Print computation time info
	print(f"Computed {num_energy_points * num_param_points} points in parallel using {num_cores} cores")
	
	return fig, ax, transmission

# Define ranges
energy_min, energy_max = 0.47, 0.475
param_min, param_max = 1.45, 1.55

# Create the plot
fig, ax, transmission_data = plot_transmission_density(
	(energy_min, energy_max),
	(param_min, param_max),
	num_energy_points = 200,
	num_param_points = 100,
	log_scale = True  # Set to True if transmission values span many orders of magnitude
)