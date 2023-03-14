"""
Generates samples from the joint mass distribution.
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from extrapops.mass import (
    sample_m1_mix_reject,
    sample_m1_invcdf,
    pdf_m1,
    pdf_m1_m2,
    sample_m2_cond_m1_rejection,
    _default_mass_params,
)

Ndraws = 1000000
nbins = 100

print("Generating %d events..." % Ndraws)
start_mix = time()
m1_sample_mix = sample_m1_mix_reject(Ndraws)
end_mix = time()
print("Time m1 mix+reject:", end_mix - start_mix)
start_invcdf = time()
m1_sample_invcdf = sample_m1_invcdf(Ndraws)
end_invcdf = time()
print("Time m1 invCDF (incl. caching):", end_invcdf - start_invcdf)
start_invcdf = time()
m1_sample_invcdf_cached = sample_m1_invcdf(Ndraws)
end_invcdf = time()
print("Time m1 invCDF (using cached):", end_invcdf - start_invcdf)

m_log10range = np.log10(_default_mass_params["m_range"])
n_m_decades = m_log10range[1] - m_log10range[0]
m1s = np.logspace(
    m_log10range[0],
    m_log10range[1],
    int(np.ceil(_default_mass_params["m_perdecade"] * n_m_decades)),
)

plt.figure(figsize=(15, 5))
plt.plot(m1s, pdf_m1(m1s), label=r"$P(m_1)$")
plt.hist(m1_sample_mix, density=True, bins=nbins, histtype="step", label="mix sample")
plt.hist(
    m1_sample_invcdf, density=True, bins=nbins, histtype="step", label="invCDF sample"
)
plt.hist(
    m1_sample_invcdf_cached,
    density=True,
    bins=nbins,
    histtype="step",
    label="invCDF sample (uses cache)",
)
plt.legend()
plt.title("m1 sample")
# plt.loglog()
plt.show(block=False)

# m2 #####################################################################################

# Generate a m2 test sample
m1_sample = m1_sample_invcdf_cached
start = time()
m2_sample = sample_m2_cond_m1_rejection(m1_sample)
end = time()
print("Time m2 reject:", end - start)

# Marginal pdf's for testing
print("Marginalising priors for testing... (this may take a while)")
m_min, m_max = _default_mass_params["m_range"]
# To avoid integrating a long null tail, cut the upper bound
m_max_int = 0.99 * m_max
ms_marg = np.logspace(np.log10(m_min), np.log10(m_max_int), 200, base=10)
eps = 1e-5
print("(marginalising m2)")
pdf_m2_marg_m1 = np.array([
    integrate.quad(lambda m1: pdf_m1_m2(m1, m2), m_min + eps, m_max_int, limit=1000)[0]
    for m2 in ms_marg])
print("(marginalising m1)")
pdf_m1_marg_m2 = np.array([
    integrate.quad(lambda m2: pdf_m1_m2(m1, m2), m_min + eps, m_max_int, limit=1000)[0]
    for m1 in ms_marg])

# Contour plot in m1,m2 -- sample
plt.figure()
nbins_2d = 500
bins = np.logspace(np.log10(m_min), np.log10(m_max), nbins_2d)
plt.hist2d(m1_sample, m2_sample, cmap="jet", bins=(bins, bins), density=True)
plt.loglog()
plt.colorbar()
plt.xlabel(r"$m_1$", fontsize=15)
plt.ylabel(r"$m_2$", fontsize=15)
plt.title("Mass Distribution Function (MC sample)", fontsize=15)
plt.show(block=False)
# Contour plot in m1,m2 -- analytic
ms = np.linspace(m_min, m_max, nbins_2d)
X, Y = np.meshgrid(ms, ms)
# NB: transform to q
Z = np.nan_to_num(pdf_m1_m2(X, Y), nan=0)
plt.figure()
plt.contourf(X, Y, Z, cmap="jet", levels=100)
plt.loglog()
plt.xlabel("m1")
plt.ylabel("m2")
plt.colorbar()
plt.title("Mass Distribution Function (analytic)", fontsize=15)
plt.show(block=False)
# Plot marg m1 dist
nbins_1d = 500
ms_plot = np.logspace(np.log10(m_min), np.log10(m_max), 500, base=10)
plt.figure()
plt.hist(
    m1_sample,
    bins=np.logspace(np.log10(m_min), np.log10(m_max), nbins_1d),
    density=True,
    label=r"$m_1$ samples",
)
plt.plot(ms_plot, pdf_m1(ms_plot), label=r"$\pi(m_1)$")
plt.plot(ms_marg, pdf_m1_marg_m2, ":", label=r"$\int\pi(m_1, m_2)\,\mathrm{d}m_2$")
plt.title("Prior m1 (marg m2)")
plt.legend()
plt.loglog()
plt.ylim(1e-4, 0.2)
plt.show(block=False)
# Plot marg m2 dist
plt.figure()
plt.hist(
    m2_sample,
    bins=np.logspace(np.log10(m_min), np.log10(m_max), nbins_1d),
    density=True,
    label=r"$m_2$ samples",
)
plt.plot(ms_marg, pdf_m2_marg_m1, label=r"$\int\pi(m_1, m_2)\,\mathrm{d}m_1$")
plt.title("Prior m2 (marg m1)")
plt.loglog()
plt.ylim(1e-6, 0.2)
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(ms_marg, pdf_m1_marg_m2, label="analytic")
plt.hist(
    m1_sample,
    bins=np.logspace(np.log10(m_min), np.log10(m_max), 1000),
    density=True,
    label="MC",
)
plt.title("Prior m1 (marg m2)")
plt.loglog()
plt.legend()
plt.ylim(1e-4, 1)
# plt.xlim(5e0, 1e1)
plt.show(block=False)

plt.figure()
plt.plot(ms_marg, pdf_m2_marg_m1, label="analytic")
plt.hist(
    m2_sample,
    bins=np.logspace(np.log10(m_min), np.log10(m_max), 1000),
    density=True,
    label="MC",
)
plt.title("Prior m2 (marg m1)")
plt.loglog()
plt.legend()
plt.ylim(1e-4, 1)
# plt.xlim(5e0, 1e1)
plt.show(block=True)
