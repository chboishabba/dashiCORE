# Minimal DASHI–LES sanity test (1D), executed

import numpy as np
import matplotlib.pyplot as plt

# domain
N = 200
x = np.linspace(0, 2*np.pi, N)

# raw field: crest + trough
u = np.sin(x) * np.exp(-0.2*(x-np.pi)**2)

# baseline smoothing
def smooth(u, k=9):
    w = np.ones(k) / k
    return np.convolve(u, w, mode="same")

u0 = smooth(u)

# signed lift
X = (u - u0) / (np.max(np.abs(u - u0)) + 1e-9)
X = np.clip(X, -1, 1)

# ternary quantisation
tau = 0.25
def ternary(X, tau):
    s = np.zeros_like(X, dtype=int)
    s[X >=  tau] = +1
    s[X <= -tau] = -1
    return s

s0 = ternary(X, tau)

# kernel closure: signed majority
def K_majority(s, radius=1):
    s_next = np.zeros_like(s)
    for i in range(len(s)):
        lo = max(0, i-radius)
        hi = min(len(s), i+radius+1)
        acc = np.sum(s[lo:hi])
        if acc > 0:   s_next[i] = +1
        elif acc < 0: s_next[i] = -1
        else:         s_next[i] = 0
    return s_next

def saturate(s, max_iter=20):
    for _ in range(max_iter):
        s_next = K_majority(s)
        if np.all(s_next == s):
            break
        s = s_next
    return s

s_star = saturate(s0)

# support × sign
m = (s_star != 0).astype(int)
sigma = s_star
assert np.all(s_star == m * sigma)

# involution test
u_neg = -u
u0_neg = smooth(u_neg)
X_neg = np.clip((u_neg - u0_neg) / (np.max(np.abs(u_neg - u0_neg)) + 1e-9), -1, 1)
s_neg = saturate(ternary(X_neg, tau))
assert np.all(s_neg == -s_star)
assert np.all((s_neg != 0) == (s_star != 0))

# structure-aware dissipation gating (LES-style)
nu_base = np.ones_like(u) * 0.1
alpha = 0.8
nu_t = nu_base * (1 - alpha * m)
assert np.all(nu_t >= 0)

# Plot results
fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

axs[0].plot(x, u, label="u (raw)")
axs[0].plot(x, u0, "--", label="baseline")
axs[0].legend()

axs[1].plot(x, X, label="signed anomaly X")
axs[1].axhline(+tau, color="k", ls=":")
axs[1].axhline(-tau, color="k", ls=":")
axs[1].legend()

axs[2].step(x, s_star, where="mid", label="s* (ternary)")
axs[2].legend()

axs[3].step(x, m, where="mid", label="support mask m")
axs[3].legend()

axs[4].plot(x, nu_t, label="gated viscosity ν_t")
axs[4].legend()

plt.tight_layout()
plt.show()
