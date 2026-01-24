import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log2

# -----------------------------
# Helpers
# -----------------------------
def smooth_1d(u, k=9):
    w = np.ones(k) / k
    return np.convolve(u, w, mode="same")

def ternary_sym(X, tau):
    s = np.zeros_like(X, dtype=np.int8)
    s[X >=  tau] = +1
    s[X <= -tau] = -1
    return s

def K_majority_1d(s, radius=1):
    s_next = np.zeros_like(s)
    n = len(s)
    for i in range(n):
        lo = max(0, i-radius)
        hi = min(n, i+radius+1)
        acc = int(np.sum(s[lo:hi]))
        if acc > 0: s_next[i] = +1
        elif acc < 0: s_next[i] = -1
        else: s_next[i] = 0
    return s_next

def saturate_1d(s, max_iter=50):
    for _ in range(max_iter):
        s_next = K_majority_1d(s)
        if np.array_equal(s_next, s):
            break
        s = s_next
    return s

def smooth2d(a, k=9):
    # separable box filter
    w = np.ones(k) / k
    tmp = np.apply_along_axis(lambda r: np.convolve(r, w, mode="same"), 1, a)
    out = np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0, tmp)
    return out

def K_majority_2d(s, radius=1):
    # s: (H,W) int8
    H, W = s.shape
    out = np.zeros_like(s)
    for i in range(H):
        i0 = max(0, i-radius)
        i1 = min(H, i+radius+1)
        for j in range(W):
            j0 = max(0, j-radius)
            j1 = min(W, j+radius+1)
            acc = int(np.sum(s[i0:i1, j0:j1]))
            if acc > 0: out[i, j] = +1
            elif acc < 0: out[i, j] = -1
            else: out[i, j] = 0
    return out

def saturate_2d(s, max_iter=25, radius=1):
    for _ in range(max_iter):
        s_next = K_majority_2d(s, radius=radius)
        if np.array_equal(s_next, s):
            break
        s = s_next
    return s

def central_diff_x(a):
    return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / 2.0

def central_diff_y(a):
    return (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / 2.0

# -----------------------------
# 1) 1D test with weak noise background
# -----------------------------
np.random.seed(7)
N = 200
x = np.linspace(0, 2*np.pi, N)
u = np.sin(x) * np.exp(-0.2*(x-np.pi)**2)
u += 0.05 * np.random.randn(N)  # weak incoherent background

u0 = smooth_1d(u, k=11)
X = u - u0
X = X / (np.max(np.abs(X)) + 1e-9)
X = np.clip(X, -1, 1)

tau = 0.25
s0 = ternary_sym(X, tau)
s_star = saturate_1d(s0, max_iter=50)
m = (s_star != 0).astype(np.uint8)
sigma = s_star

# involution sanity
u_neg = -u
u0_neg = smooth_1d(u_neg, k=11)
X_neg = np.clip((u_neg - u0_neg) / (np.max(np.abs(u_neg - u0_neg)) + 1e-9), -1, 1)
s_neg = saturate_1d(ternary_sym(X_neg, tau), max_iter=50)
assert np.all(s_neg == -s_star)
assert np.all((s_neg != 0) == (s_star != 0))
assert np.all(s_star == m * sigma)

nu_base_1d = np.ones_like(u) * 0.1
alpha = 0.8
nu_t_1d = nu_base_1d * (1 - alpha * m)

# Plots (one figure each; no subplots)
plt.figure(figsize=(10,3))
plt.plot(x, u, label="u (raw+noise)")
plt.plot(x, u0, "--", label="baseline")
plt.legend()
plt.title("1D: raw field and baseline")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,3))
plt.plot(x, X, label="signed anomaly X")
plt.axhline(+tau, color="k", ls=":")
plt.axhline(-tau, color="k", ls=":")
plt.legend()
plt.title("1D: signed anomaly and thresholds")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,2.5))
plt.step(x, s_star, where="mid", label="s* (ternary)")
plt.ylim(-1.2, 1.2)
plt.legend()
plt.title("1D: saturated ternary structure s*")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,2.5))
plt.plot(x, nu_t_1d, label="gated viscosity ν_t")
plt.legend()
plt.title("1D: LES-style gated viscosity")
plt.tight_layout()
plt.show()

# -----------------------------
# 2) 2D vortex pair + signed vorticity ωz
# -----------------------------
H, W = 128, 128
yy, xx = np.mgrid[0:H, 0:W]
x0, y0 = W*0.35, H*0.5
x1, y1 = W*0.65, H*0.5
sig = 10.0

# Streamfunction for two opposite vortices
psi = np.exp(-((xx-x0)**2 + (yy-y0)**2)/(2*sig**2)) - np.exp(-((xx-x1)**2 + (yy-y1)**2)/(2*sig**2))

# Velocity from streamfunction: u = dψ/dy, v = -dψ/dx
u2 = central_diff_y(psi)
v2 = -central_diff_x(psi)

# Vorticity ωz = dv/dx - du/dy
omega = central_diff_x(v2) - central_diff_y(u2)

# Baseline + signed lift on ω
omega0 = smooth2d(omega, k=9)
Xw = omega - omega0
Xw = Xw / (np.max(np.abs(Xw)) + 1e-9)
Xw = np.clip(Xw, -1, 1)

tau2 = 0.30
s0_2d = ternary_sym(Xw, tau2)
s_star_2d = saturate_2d(s0_2d, max_iter=20, radius=1)
m2 = (s_star_2d != 0).astype(np.uint8)

# Plots (one figure each)
plt.figure(figsize=(5,5))
plt.imshow(omega, origin="lower")
plt.title("2D: vorticity ωz")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(Xw, origin="lower")
plt.title("2D: signed anomaly X(ωz)")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(s_star_2d, origin="lower", vmin=-1, vmax=1)
plt.title("2D: saturated ternary structure s* (ωz)")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(m2, origin="lower", vmin=0, vmax=1)
plt.title("2D: support mask m")
plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------
# 3) Dynamic SGS toy: nu_base from |∇u| and gate it
# -----------------------------
# Use magnitude of velocity gradient as crude "strain proxy"
dux_dx = central_diff_x(u2)
dux_dy = central_diff_y(u2)
dvy_dx = central_diff_x(v2)
dvy_dy = central_diff_y(v2)

# |grad u| Frobenius norm
grad_mag = np.sqrt(dux_dx**2 + dux_dy**2 + dvy_dx**2 + dvy_dy**2)

# Smag-like base viscosity (toy): nu_base ∝ Δ^2 * |grad u|
Delta = 1.0
Cs = 0.15
nu_base = (Cs * Delta)**2 * grad_mag

alpha_dyn = 0.7
nu_t = nu_base * (1 - alpha_dyn * m2)  # suppress in coherent signed structure
nu_t = np.maximum(0.0, nu_t)

plt.figure(figsize=(5,5))
plt.imshow(nu_base, origin="lower")
plt.title("2D: toy ν_base from |∇u|")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(nu_t, origin="lower")
plt.title("2D: gated ν_t (structure-aware)")
plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------
# 4) Tsunami corridor mock: ray density through random smooth bathymetry
# -----------------------------
# Build smooth random speed field c(x,y) = 1 + eps * smooth_noise
np.random.seed(3)
Hc, Wc = 160, 240
noise = np.random.randn(Hc, Wc)
c = 1.0 + 0.12 * smooth2d(noise, k=21)   # smooth, weak variations
# clip to keep positive and modest
c = np.clip(c, 0.7, 1.3)

# Precompute gradients of c
dc_dx = central_diff_x(c)
dc_dy = central_diff_y(c)

# Ray integration for Hamiltonian H = c(x)*|k| with |k|~1:
# xdot = c * k/|k|, kdot = -∇c  (with |k| held ~1 by normalization)
num_rays = 5000
steps = 600
dt = 0.6

# initial positions near left edge, centered in y
y0 = Hc * 0.5 + 5.0*np.random.randn(num_rays)
x0 = np.ones(num_rays) * 10.0

# initial directions: mostly to the right with small angular spread
theta = 0.0 + 0.15*np.random.randn(num_rays)
kx = np.cos(theta)
ky = np.sin(theta)

xr = x0.copy()
yr = y0.copy()

# density accumulation
dens = np.zeros((Hc, Wc), dtype=np.float64)

for _ in range(steps):
    # sample c and grad at ray positions (nearest-neighbor for simplicity)
    xi = np.clip(np.rint(xr).astype(int), 0, Wc-1)
    yi = np.clip(np.rint(yr).astype(int), 0, Hc-1)

    ci = c[yi, xi]
    gx = dc_dx[yi, xi]
    gy = dc_dy[yi, xi]

    # advance position
    xr += dt * ci * kx
    yr += dt * ci * ky

    # advance direction (bend)
    kx -= dt * gx
    ky -= dt * gy
    # renormalize direction
    kn = np.sqrt(kx*kx + ky*ky) + 1e-12
    kx /= kn
    ky /= kn

    # accumulate density for rays still in bounds
    inb = (xr >= 0) & (xr < Wc) & (yr >= 0) & (yr < Hc)
    xi2 = np.clip(np.rint(xr[inb]).astype(int), 0, Wc-1)
    yi2 = np.clip(np.rint(yr[inb]).astype(int), 0, Hc-1)
    np.add.at(dens, (yi2, xi2), 1.0)

# Normalize density and compute signed anomaly relative to smooth baseline
dens0 = smooth2d(dens, k=31)
Xd = dens - dens0
Xd = Xd / (np.max(np.abs(Xd)) + 1e-9)
Xd = np.clip(Xd, -1, 1)

tau_d = 0.22
s0_d = ternary_sym(Xd, tau_d)
s_star_d = saturate_2d(s0_d, max_iter=15, radius=1)
m_d = (s_star_d != 0).astype(np.uint8)

plt.figure(figsize=(7,4.5))
plt.imshow(dens, origin="lower")
plt.title("Tsunami mock: ray density (branched flow corridors)")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
plt.imshow(Xd, origin="lower")
plt.title("Tsunami mock: signed anomaly X(density)")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
plt.imshow(s_star_d, origin="lower", vmin=-1, vmax=1)
plt.title("Tsunami mock: saturated ternary structure s*")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
plt.imshow(m_d, origin="lower", vmin=0, vmax=1)
plt.title("Tsunami mock: support mask m (corridor band seeds)")
plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------
# 5) MDL ledger toy: signed vs unsigned skeleton + residual
# -----------------------------
def skeleton_recon(X, s_star, k_smooth=11):
    # Keep only values where s_star != 0 (signed), then smooth to form chart X_hat
    X_keep = np.zeros_like(X, dtype=np.float64)
    X_keep[s_star != 0] = X[s_star != 0]
    X_hat = smooth2d(X_keep, k=k_smooth) if X_hat_needed_2d(X) else smooth_1d(X_keep, k=k_smooth)
    return X_hat

def X_hat_needed_2d(X):
    return (X.ndim == 2)

def skeleton_recon_signed(X, s_star, k_smooth=11):
    X_keep = np.zeros_like(X, dtype=np.float64)
    X_keep[s_star != 0] = X[s_star != 0]  # signed values kept
    X_hat = smooth2d(X_keep, k=k_smooth) if X.ndim == 2 else smooth_1d(X_keep, k=k_smooth)
    return X_hat

def skeleton_recon_unsigned(X, s_star, k_smooth=11):
    # Drop negative structure: keep only positive support
    X_keep = np.zeros_like(X, dtype=np.float64)
    X_keep[s_star == +1] = X[s_star == +1]
    X_hat = smooth2d(X_keep, k=k_smooth) if X.ndim == 2 else smooth_1d(X_keep, k=k_smooth)
    return X_hat

def mdl_ledger(X, s_star, band_mask, unsigned=False):
    # Very simple ledger:
    # - skeleton bits: (#nonzero indices)*log2(|G|) + (#nonzero)*1(sign) (sign bit omitted for unsigned)
    # - residual bits: n_band * log2(std/eps)  (proxy for entropy)
    # and we report residual MSE on band too.
    G = X.size
    nnz = int(np.sum(s_star != 0))
    nnz_pos = int(np.sum(s_star == +1))
    nnz_neg = int(np.sum(s_star == -1))
    n_band = int(np.sum(band_mask))
    eps = 1e-6

    if unsigned:
        idx_bits = nnz_pos * log2(G + 1e-12)
        sign_bits = 0.0
        X_hat = skeleton_recon_unsigned(X, s_star, k_smooth=21)
    else:
        idx_bits = nnz * log2(G + 1e-12)
        sign_bits = nnz * 1.0
        X_hat = skeleton_recon_signed(X, s_star, k_smooth=21)

    resid = (X - X_hat)
    resid_band = resid[band_mask.astype(bool)]
    mse_band = float(np.mean(resid_band**2)) if n_band > 0 else float(np.mean(resid**2))
    std_band = float(np.std(resid_band)) if n_band > 0 else float(np.std(resid))

    # entropy-ish proxy
    resid_bits = n_band * max(0.0, log2(std_band/eps + 1.0))
    total_bits = idx_bits + sign_bits + resid_bits

    return {
        "G": G,
        "nnz_total": nnz,
        "nnz_pos": nnz_pos,
        "nnz_neg": nnz_neg,
        "band_cells": n_band,
        "skeleton_bits": idx_bits + sign_bits,
        "residual_bits": resid_bits,
        "total_bits": total_bits,
        "band_mse": mse_band,
        "band_resid_std": std_band,
    }

# Band: 1-step dilation of support (simple)
def band_from_support(m, radius=1):
    if m.ndim == 1:
        b = m.copy().astype(bool)
        for _ in range(radius):
            b = b | np.roll(b, 1) | np.roll(b, -1)
        return b.astype(np.uint8)
    else:
        b = m.copy().astype(bool)
        for _ in range(radius):
            b = b | np.roll(b, 1, axis=0) | np.roll(b, -1, axis=0) | np.roll(b, 1, axis=1) | np.roll(b, -1, axis=1)
        return b.astype(np.uint8)

band_d = band_from_support(m_d, radius=2)

signed_stats = mdl_ledger(Xd, s_star_d, band_d, unsigned=False)
unsigned_stats = mdl_ledger(Xd, s_star_d, band_d, unsigned=True)

df = pd.DataFrame([
    {"variant": "signed (+/- skeleton)", **signed_stats},
    {"variant": "unsigned (+ only)", **unsigned_stats},
])

# Display summary
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("DASHI–LES MDL ledger (toy) — signed vs unsigned", df)

# Print a small human summary
print("MDL ledger (toy, tsunami mock):")
print(df[["variant","total_bits","skeleton_bits","residual_bits","band_mse","band_resid_std","nnz_pos","nnz_neg","band_cells"]])
