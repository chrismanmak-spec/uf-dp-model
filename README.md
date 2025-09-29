# uf/dp model
bridging realtivity and quantum mechanics using uf/dp to brige for an attempt of gut.
python

Copy Code
# dp_uf_em_rad.py
# 2D dp/uf toy with Maxwell + grey radiation moments
# Requirements: Python 3.9+, numpy
import numpy as np
import time
from math import sqrt

# ----------------- parameters -----------------
Nx = Ny = 128
L = 1.0
dx = L / Nx
dt = 0.2 * dx  # conservative; ensure dt <= dx/c_max
nsteps = 1000

# physical (dimensionless) params
c = 10.0        # light speed (>> cuf)
cuf = 1.0
epsilon0 = 1.0
mu0 = 1.0 / epsilon0
gamma_em = 0.001
kappa0 = 1.0    # opacity prefactor
eps_emiss = 0.1
q_charge = 0.0  # set small nonzero to test Lorentz forces
alpha = 0.5
kappa_uf = 1.0  # coupling uf <- sources
gamma_uf = 0.02
# dp fluid params
cs = 0.1
A_ex = 0.5
n_exp = 1.5
Excess = 0.8
eta_S = 0.02
rho0 = 1.0
rho_c = 0.3
drho = 0.05
tau0 = 1.0
lambda_tau = 0.05
beta_tau = 0.5
tauscale = 0.2
# ----------------- grid & utility -----------------
def idx(i, N): return i % N
def roll_x(a, s): return np.roll(a, -s, axis=0)
def roll_y(a, s): return np.roll(a, -s, axis=1)

# central derivative operators
def ddx(f):
    return (roll_x(f, -1) - roll_x(f, 1)) / (2*dx)
def ddy(f):
    return (roll_y(f, -1) - roll_y(f, 1)) / (2*dx)
def lap(f):
    return (roll_x(f, -1) + roll_x(f, 1) + roll_y(f, -1) + roll_y(f, 1) - 4*f) / (dx*dx)
def div_fx_fy(fx, fy):
    return ddx(fx) + ddy(fy)

# ----------------- fields init -----------------
x = (np.arange(Nx)+0.5)*dx
y = (np.arange(Ny)+0.5)*dx
X, Y = np.meshgrid(x,y, indexing='ij')

# dp fields
rho = 0.01 + 1.0 * np.exp(-((X-0.5)**2 + (Y-0.5)**2)/(2*(0.05**2)))
vx = np.zeros((Nx,Ny)); vy = np.zeros((Nx,Ny))

# uf scalar wave (u surrogate)
u = np.zeros((Nx,Ny)); u_prev = np.zeros_like(u)
ut = np.zeros_like(u)
tau = np.ones_like(u) * tau0

# EM fields
Ex = np.zeros((Nx,Ny)); Ey = np.zeros((Nx,Ny)); Bz = np.zeros((Nx,Ny))

# radiation moments
E_rad = np.zeros((Nx,Ny)); Fx = np.zeros((Nx,Ny)); Fy = np.zeros((Nx,Ny))

# helper functions
def sigmoid(x): return 1.0/(1.0 + np.exp(-x))

# energy bookkeeping (diagnostics)
def em_energy():
    return 0.5*(epsilon0*(Ex**2 + Ey**2) + (1.0/mu0)*Bz**2)

def total_energy_u():
    eu = 0.5*(ut**2 + cuf**2 * ( (ddx(u))**2 + (ddy(u))**2 ))  # approximate grad^2
    return np.sum(eu) * dx*dx

# ----------------- time stepping -----------------
start = time.time()
for step in range(nsteps):
    # pressure and pEx
    p = cs**2 * rho + A_ex * Excess * rho**n_exp

    # compute S transfer (dp -> uf) (simple smooth form)
    S = eta_S * sigmoid((rho - rho_c)/drho) * max(0.0, Excess - 0.5) * np.exp(-tau/tauscale) * (rho / rho0)

    # Maxwell: compute current J = q * rho * v
    Jx = q_charge * rho * vx
    Jy = q_charge * rho * vy

    # Maxwell explicit update
    dBz = -(ddx(Ey) - ddy(Ex))
    Ex_new = Ex + dt * ( c*c * ddy(Bz) - Jx/epsilon0 - gamma_em * Ex )
    Ey_new = Ey + dt * ( -c*c * ddx(Bz) - Jy/epsilon0 - gamma_em * Ey )
    Bz_new = Bz + dt * ( dBz - gamma_em * Bz )

    Ex, Ey, Bz = Ex_new, Ey_new, Bz_new

    # EM stress-energy (simple)
    E2 = Ex*Ex + Ey*Ey
    B2 = Bz*Bz
    TEM_00 = 0.5*(epsilon0*E2 + (1.0/mu0)*B2)
    # Poynting/momentum density approx
    TEM_x0 = (1.0/mu0) * (Ey * Bz)
    TEM_y0 = -(1.0/mu0) * (Ex * Bz)

    # Radiation emissivity & opacity
    eta_emiss = eps_emiss * rho * sigmoid((Excess-0.5)/0.1)
    kappa_abs = kappa0 * rho

    # Radiation update (explicit)
    divF = div_fx_fy(Fx, Fy)
    E_rad_new = E_rad + dt * ( - divF + eta_emiss - kappa_abs * c * E_rad )
    # closure: compute P_rad via simple isotropic approx or Minerbo factor
    Fx_mag = np.sqrt(Fx*Fx + Fy*Fy) + 1e-12
    f = Fx_mag / (c * (E_rad + 1e-12))
    # Minerbo-like chi
    chi = (3.0 + 4.0*f*f) / (5.0 + 2.0*np.sqrt(4.0 - 3.0*np.minimum(f*f, 0.9999)))
    # build P components
    # isotropic part
    P_xx = chi * E_rad
    P_yy = chi * E_rad
    # beam part
    ux = Fx / (Fx_mag + 1e-16); uy = Fy / (Fx_mag + 1e-16)
    P_xx += (1.0 - chi) * (Fx*Fx) / (Fx_mag*Fx_mag + 1e-16)
    P_xy = (1.0 - chi) * (Fx*Fy) / (Fx_mag*Fx_mag + 1e-16)
    P_yy += (1.0 - chi) * (Fy*Fy) / (Fx_mag*Fx_mag + 1e-16)

    # divergence of P
    divP_x = ddx(P_xx) + ddy(P_xy)
    divP_y = ddx(P_xy) + ddy(P_yy)
    Fx_new = Fx + dt * ( - c*c * divP_x - kappa_abs * c * Fx )
    Fy_new = Fy + dt * ( - c*c * divP_y - kappa_abs * c * Fy )

    E_rad, Fx, Fy = E_rad_new, Fx_new, Fy_new

    # dp momentum update: compute forces
    # Lorentz force per unit volume (rho_charge*(E + v x B))
    rho_charge = q_charge * rho
    fEMx = rho_charge * (Ex + vy * Bz)
    fEMy = rho_charge * (Ey - vx * Bz)
    # radiative momentum exchange (absorption drag)
    frad_x = - kappa_abs * Fx
    frad_y = - kappa_abs * Fy

    # momentum conservative update (simple explicit)
    momx = rho * vx; momy = rho * vy
    # compute advective flux divergence approx (simple)
    # here use centered fluxes for demo; replace with upwind on real runs
    momx_flux_x = ddx(momx * vx)
    momx_flux_y = ddy(momx * vy)
    momy_flux_x = ddx(momy * vx)
    momy_flux_y = ddy(momy * vy)

    momx = momx - dt * ( momx_flux_x + momx_flux_y + ddx(p) + rho * ddx(u) + gamma_uf * momx ) + dt * (fEMx + frad_x)
    momy = momy - dt * ( momy_flux_x + momy_flux_y + ddy(p) + rho * ddy(u) + gamma_uf * momy ) + dt * (fEMy + frad_y)

    # update velocities
    rho_new = rho - dt * div_fx_fy(rho*vx, rho*vy) - dt * S
    rho_new = np.maximum(rho_new, 1e-12)
    vx = momx / rho_new; vy = momy / rho_new
    rho = rho_new

    # uf scalar wave update (leapfrog-like)
    Lu = lap(u)
    u_new = 2.0*u - u_prev + dt*dt * ( cuf*cuf * Lu - gamma_uf * ut + kappa_uf * (tau**-1) * (rho + alpha * p + TEM_00) )
    ut = (u_new - u_prev) / (2.0*dt)
    u_prev = u.copy()
    u = u_new

    # tau relaxation
    tau += dt * ( -lambda_tau * (tau - tau0) + beta_tau * S )

    # energy exchange: radiation emission/absorption update dp internal energy implicitly via S and emission/abs
    # for this toy we do minimal bookkeeping, not full internal energy update

    # simple diagnostics print
    if step % 50 == 0:
        Etot_em = np.sum(em_energy())*dx*dx
        Erad = np.sum(E_rad)*dx*dx
        mass = np.sum(rho)*dx*dx
        print(f"step {step:5d} mass={mass:.4f} Em={Etot_em:.4e} Erad={Erad:.4e}")

end = time.time()
print("done, time:", end-start)
# mode_sum_Tmunu.py
# Phase‑1 NumPy prototype: mode-sum for a real scalar field T^{mu nu}
# Requires: Python 3.9+, numpy
# Optional: numba for acceleration
import numpy as np
import time
from math import sqrt

# Optional: enable numba by uncommenting these lines if numba installed
# from numba import njit, prange

# ------------------ defaults / configuration ------------------
m_default = 0.0        # field mass
xi_default = 0.0       # curvature coupling
mu_ren = 1.0           # renormalization scale for logs (EFT)
kmax_default = 200.0
Nk_default = 1000
tol_change = 1e-3      # relative change threshold to trigger callback

# ------------------ helpers ------------------
def kgrid_1d(kmax=kmax_default, Nk=Nk_default):
    """Return 1D isotropic k-grid (array) and weights for simple trapezoid."""
    k = np.linspace(0.0, kmax, Nk)
    dk = k[1]-k[0]
    w = np.ones_like(k)*dk
    w[0] = dk*0.5
    w[-1] = dk*0.5
    return k, w

def bose_occupation(k, T):
    """Thermal occupation for massless modes (omega = k); T in same units."""
    omega = k
    with np.errstate(divide='ignore', invalid='ignore'):
        n = 1.0 / (np.exp(omega / (T + 1e-16)) - 1.0)
    n[omega==0] = 0.0
    return n

# ------------------ mode evolution (homogeneous background) ------------------
def init_mode_amplitudes(k, m=m_default, state_params=None):
    """
    Initialize mode amplitudes and time-derivatives for each k.
    state_params can include:
      - 'type': 'vacuum'|'thermal'|'bogoliubov'|'custom'
      - 'T': temperature for thermal
      - 'beta_k': array of Bogoliubov beta coefficients (complex) same length as k
      - 'n_k': occupation numbers array (real)
    Returns arrays phi_k (complex), phi_k_dot (complex).
    """
    omega = np.sqrt(k*k + m*m)
    Nk = len(k)
    phi = np.zeros(Nk, dtype=np.complex128)
    phidot = np.zeros(Nk, dtype=np.complex128)

    stype = 'vacuum' if state_params is None else state_params.get('type','vacuum')
    if stype == 'vacuum':
        phi[:] = 1.0/np.sqrt(2.0*omega)
        phidot[:] = -1j * omega * phi
    elif stype == 'thermal':
        T = state_params.get('T', 1.0)
        n_k = bose_occupation(k, T)
        # Random-phase ensemble producing correct 2-point function: sample amplitude sqrt(n+1/2)
        amp = np.sqrt(n_k + 0.5)
        phases = np.exp(1j * 2*np.pi * np.random.rand(Nk))
        phi[:] = amp * phases / np.sqrt(omega)
        phidot[:] = -1j * omega * phi
    elif stype == 'bogoliubov':
        beta = state_params.get('beta_k')
        if beta is None:
            raise ValueError("bogoliubov state requires 'beta_k' array")
        alpha = np.sqrt(1.0 + np.abs(beta)**2)
        phi_vac = 1.0/np.sqrt(2.0*omega)
        phi[:] = alpha*phi_vac + np.conjugate(beta)*np.conjugate(phi_vac)
        phidot[:] = -1j*omega*(alpha*phi_vac - np.conjugate(beta)*np.conjugate(phi_vac))
    elif stype == 'custom':
        phi[:] = state_params.get('phi_k', np.ones(Nk,dtype=np.complex128))
        phidot[:] = state_params.get('phidot_k', -1j*np.sqrt(k*k+m*m)*phi)
    else:
        raise ValueError("unknown state type")
    return phi, phidot

def evolve_modes_rk2(phi, phidot, k, dt, m=m_default, a=1.0, adot=0.0):
    """
    Simple RK2 (midpoint) evolve homogeneous modes for one time step.
    Equation for conformal time variable would differ; here assume flat-space-like:
      phi_ddot + omega_k^2 * phi = 0, with omega_k^2 = k^2 + m^2 (a-dependence can be added).
    For weakly time-dependent scale factor, use minimal corrections via a, adot params.
    """
    omega2 = k*k + (a*m)**2
    # RHS: second derivative
    def accel(phi_): return -omega2 * phi_
    phi_mid = phi + 0.5*dt*phidot
    phidot_mid = phidot + 0.5*dt*accel(phi)
    phi_new = phi + dt * phidot_mid
    phidot_new = phidot + dt * accel(phi_mid)
    return phi_new, phidot_new

# ------------------ adiabatic subtraction terms (up to 4th order) ------------------
def adiabatic_subtractions(k, m=0.0, xi=0.0, order=4, a=1.0, adot=0.0, addot=0.0):
    """
    Return subtraction pieces for energy density and pressures for each k.
    We implement standard WKB/adiabatic expansions in flat-ish background.
    For simplicity, this prototype returns the leading terms that cancel UV divergences:
      - energy density subtraction per mode: omega/2 + order-2 corrections
    Note: full 4th-order expressions include many terms; here we include up to 2nd
    order analytic pieces and a log regulator matching prescription for 4th order.
    This is a pragmatic, working prototype; extend if you need exact DeWitt-4 terms.
    """
    omega = np.sqrt(k*k + (a*m)**2)
    # Leading vacuum: 1/2 * omega
    sub_rho = 0.5*omega
    # 2nd order: ( (xi-1/6) R )/(4 omega) ... for homogeneous a(t) simplified
    # Here approximate R ~ 6*(addot/a + (adot/a)^2); include minimal correction
    R = 6.0 * (addot/a + (adot/a)**2)
    sub_rho += ( (xi - 1.0/6.0) * R ) / (4.0 * omega + 1e-30)
    # 4th-order log term (schematic): ~ (m^4) / (32 omega^5) * log(mu)
    # Use EFT-style subtraction: remove small remainder by subtracting a regulated log tail
    if order >= 4:
        # construct a smooth regulator that damps high-k tail
        sub_rho += (m**4) / (32.0 * (omega**5 + 1e-30)) * np.log((omega + 1e-16) / (mu_ren + 1e-16))
    # For pressure components in isotropic homogeneous case, P = (k^2/(3 omega)) * (1/2) + corrections
    sub_P = 0.5 * (k*k) / (3.0 * omega + 1e-30)
    sub_P += ( (xi - 1.0/6.0) * R ) / (12.0 * omega + 1e-30)
    return sub_rho, sub_P

# ------------------ compute bare mode-sum T00 and pressures ------------------
def mode_sum_Tmunu(k, w, phi, phidot, m=0.0, xi=0.0, a=1.0, adot=0.0, addot=0.0):
    """
    Compute bare energy density and pressure from mode amplitudes.
    Returns arrays per mode; caller integrates with weights to get spatially-averaged values.
    For homogeneous background, T^0_0 = energy density; T^i_i = 3 P (trace components).
    """
    omega = np.sqrt(k*k + (a*m)**2)
    # mode energy density per k (bare)
    # E_k = 0.5*(|phidot|^2 + omega^2 * |phi|^2)
    E_k = 0.5 * (np.abs(phidot)**2 + (omega**2) * np.abs(phi)**2)
    # momentum-squared expectation gives pressure: P_k = (k^2/(3 omega^2)) * E_k approx
    P_k = (k*k) / (3.0*(omega**2 + 1e-30)) * E_k
    return E_k, P_k

# ------------------ public API: compute_Tmunu ------------------
class ChangeReporter:
    """Simple object to hold previous integrated energy and call callback on large changes."""
    def __init__(self, threshold=tol_change, callback=None):
        self.prev_energy = None
        self.threshold = threshold
        self.callback = callback

    def check(self, energy, info=None):
        if self.prev_energy is None:
            self.prev_energy = energy
            return False
        rel = abs(energy - self.prev_energy) / (abs(self.prev_energy) + 1e-30)
        triggered = rel > self.threshold
        if triggered and self.callback is not None:
            # send simple report
            self.callback({
                'type': 'Tmunu_large_change',
                'prev_energy': self.prev_energy,
                'new_energy': energy,
                'rel_change': rel,
                'info': info
            })
        self.prev_energy = energy
        return triggered

def compute_Tmunu(background, state_params=None, kparams=None, renorm_order=4,
                  reporter=None, dt_evolve=0.0, n_steps=1):
    """
    Main function to compute spatially-averaged renormalized <T^mu_nu> for homogeneous background.
    background: dict with keys:
       - 'a': scale factor (float) or callable a(t); default 1.0
       - 'adot': time derivative (float) or callable; default 0.0
       - 'addot': second derivative (float) or callable; default 0.0
       - 'm': mass (float)
       - 'xi': coupling
    state_params: see init_mode_amplitudes
    kparams: dict {'kmax':, 'Nk':}
    reporter: ChangeReporter instance to detect/report large changes
    dt_evolve, n_steps: optional small-mode evolution parameters (for time-dep backgrounds)
    Returns: dict with keys 'rho' (energy density), 'P' (pressure), 'raw' (mode arrays)
    """
    m = background.get('m', m_default)
    xi = background.get('xi', xi_default)
    a = background.get('a', 1.0)
    adot = background.get('adot', 0.0)
    addot = background.get('addot', 0.0)

    kmax = kparams.get('kmax', kmax_default) if kparams else kmax_default
    Nk = kparams.get('Nk', Nk_default) if kparams else Nk_default
    k, w = kgrid_1d(kmax=kmax, Nk=Nk)

    # init modes
    phi, phidot = init_mode_amplitudes(k, m=m, state_params=state_params)

    # optional evolution steps (simple RK2) to let modes respond to background
    for _ in range(n_steps):
        phi, phidot = evolve_modes_rk2(phi, phidot, k, dt_evolve, m=m, a=a, adot=adot)

    E_k, P_k = mode_sum_Tmunu(k, w, phi, phidot, m=m, xi=xi, a=a, adot=adot, addot=addot)

    # subtract adiabatic pieces per mode
    sub_rho_k, sub_P_k = adiabatic_subtractions(k, m=m, xi=xi, order=renorm_order, a=a, adot=adot, addot=addot)

    # integrate over k: 4*pi * integral k^2 dk (for 3D isotropic), but our kgrid is 1D representing radial k
    # so measure factor is 4*pi*k^2. For prototypes that use 1D k-sum, include measure here.
    measure = 4.0 * np.pi * k*k
    integrand_rho = (E_k - sub_rho_k) * measure
    integrand_P = (P_k - sub_P_k) * measure

    # integrate using weights w (k-grid)
    rho = np.sum(integrand_rho * w)
    P = np.sum(integrand_P * w)

    # small-k handling: remove k=0 singular contribution by forcing regular behavior
    if k[0] == 0.0:
        # replace first term with analytic small-k limit (for massless, vacuum -> 0)
        integrand0_rho = 0.0
        integrand0_P = 0.0
        rho -= ( (E_k[0] - sub_rho_k[0]) * measure[0] * w[0] )
        P -= ( (P_k[0] - sub_P_k[0]) * measure[0] * w[0] )
        rho += integrand0_rho
        P += integrand0_P

    # Final unit factors: for homogeneous average, rho and P are per-volume (no extra dx)
    result = {'rho': rho, 'P': P, 'k': k, 'E_k': E_k, 'P_k': P_k, 'sub_rho_k': sub_rho_k, 'sub_P_k': sub_P_k}

    # reporter check: integrated energy
    if reporter is not None:
        reporter.check(rho, info={'P': P, 'kmax': kmax, 'Nk': Nk})

    return result

# ------------------ integration example: hooking into dp/uf main loop ------------------
def example_integration_loop(steps=10):
    """
    Minimal demo of calling compute_Tmunu each main step and reporting large changes.
    """
    reporter = ChangeReporter(threshold=tol_change, callback=lambda rpt: print("REPORT:", rpt))
    background = {'a':1.0, 'adot':0.0, 'addot':0.0, 'm':m_default, 'xi':xi_default}
    state = {'type':'vacuum'}
    for i in range(steps):
        # optionally modify state or background slowly to emulate dynamics
        # e.g., small periodic perturbation:
        background['a'] = 1.0 + 0.001 * np.sin(0.1 * i)
        res = compute_Tmunu(background, state_params=state, kparams={'kmax':50.0,'Nk':800}, reporter=reporter,
                             renorm_order=4, dt_evolve=0.0, n_steps=0)
        print(f"step {i:3d} rho={res['rho']:.6e}  P={res['P']:.6e}")

if __name__ == "__main__":
    # quick smoke test
    t0 = time.time()
    example_integration_loop(steps=6)
    t1 = time.time()
    print("done, time:", t1-t0)


# params.py
# Unified configuration for the dp/uf coupled solver

# --- Visualization & I/O ---
VIS_FRAMERATE = 50           # Save a plot every N steps
OUTPUT_DIR = "simulation_frames" # Directory to save frames

# --- Mock EFE Solver Parameters ---
G_NEWTON = 6.674e-2  # Mock gravitational constant (tweak for effect)
H0 = 0.1             # Mock Hubble parameter for initial expansion
FRIEDMANN_DAMPING = 0.01 # A small damping term for stability

# --- Numerical Grid Parameters ---
Nx = 128
Ny = 128
L = 1.0
dx = L / Nx
dt = 0.1 * dx  # Reduced dt for stability with new physics
nsteps = 1000

# --- QFTCS / UF Field Parameters ---
m_default = 0.0      # field mass
xi_default = 0.0     # curvature coupling
mu_ren = 1.0         # renormalization scale for logs (EFT)
kmax_default = 200.0 # max k for mode sum
Nk_default = 1000    # number of k points
tol_change = 1e-3    # change threshold for QFTCS reporter
cuf = 1.0            # uf wave speed
kappa_uf = 1.0       # coupling uf <- sources
gamma_uf = 0.02      # uf damping/friction

# --- DP Hydro / EM / Radiation Parameters ---
c = 10.0             # light speed (must be >> cuf)
epsilon0 = 1.0
mu0 = 1.0 / epsilon0
gamma_em = 0.001     # EM damping
eps_emiss = 0.1      # Radiation emissivity
kappa0 = 1.0         # Radiation opacity prefactor
q_charge = 0.0       # Charge factor
alpha = 0.5          # Term for pressure contribution in uf source

# --- DP Fluid / Transfer (S) Parameters ---
cs = 0.1             # sound speed
A_ex = 0.5           # Excess amplitude factor
n_exp = 1.5          # Density exponent
Excess = 0.8         # Initial dp Surplus proxy
eta_S = 0.02         # S transfer rate prefactor
rho0 = 1.0
rho_c = 0.3          # Critical density for S transfer
drho = 0.05
tau0 = 1.0           # Initial uf tension proxy
lambda_tau = 0.05    # tau relaxation rate
beta_tau = 0.5       # tau change with S transfer
tauscale = 0.2       # tau exponential damping scale
