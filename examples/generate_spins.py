"""
Generate samples from the spin distribution.
"""

import numpy as np
import matplotlib.pyplot as plt

from extrapops.spin import (
    sample_spin_amplitude,
    pdf_spin_amplitude,
    pdf_2spins_cos,
    pdf_spin_cos_marg,
    sample_2spins_cos,
)

N = 1000000

# Amplitude #############################################################################

a1_sample = sample_spin_amplitude(N)
a2_sample = sample_spin_amplitude(N)

nbins = 200
a_x = np.linspace(min(a1_sample), max(a1_sample), min(2 * nbins, 400))

# 1d
plt.figure()
plt.hist(a1_sample, bins=nbins, density=True, histtype="step")
plt.hist(a2_sample, bins=nbins, density=True, histtype="step")
plt.plot(a_x, pdf_spin_amplitude(a_x))
plt.xlabel(r"$a_1$", fontsize=15)
plt.title("Single (1d marg) spin Amplitude Distribution Function", fontsize=15)
plt.show(block=False)

# 2d histogram vs contours
plt.figure()
plt.hist2d(a1_sample, a2_sample, cmap="viridis", bins=nbins, density=True)
plt.colorbar()
# Contour plot
range_a = [0, 1]
delta = (range_a[1] - range_a[0]) / (2 * nbins)
x = np.arange(range_a[0], range_a[1], delta)
y = x.copy()
X, Y = np.meshgrid(x, y)
Z = pdf_spin_amplitude(X) * pdf_spin_amplitude(Y)
CS = plt.contour(X, Y, Z, colors="k")
plt.clabel(CS, inline=True, fontsize=10)
plt.xlabel(r"$a_1$", fontsize=15)
plt.ylabel(r"$a_2$", fontsize=15)
plt.title("Spin Amplitude Distribution Function", fontsize=15)
plt.show(block=False)

# Alignment #############################################################################

cost1_sample, cost2_sample = sample_2spins_cos(N)

# 1d marginals
plt.figure()
plt.hist(cost1_sample, bins=nbins, density=True, histtype="step")
plt.hist(cost2_sample, bins=nbins, density=True, histtype="step")
cosines = np.linspace(-1, 1, 400)
plt.plot(cosines, pdf_spin_cos_marg(cosines))
plt.title("Single (1d marg) spin alignment Distribution Function", fontsize=15)
plt.show(block=False)

# 2d histogram vs contours
plt.figure()
nbins = 200
plt.hist2d(cost1_sample, cost2_sample, cmap="viridis", bins=nbins, density=True)
plt.colorbar()
# Contour plot
range_cos = [-1, 1]
delta = (range_cos[1] - range_cos[0]) / (2 * nbins)
x = np.arange(range_cos[0], range_cos[1], delta)
y = x.copy()
X, Y = np.meshgrid(x, y)
Z = pdf_2spins_cos(X, Y)
CS = plt.contour(X, Y, Z, colors="k")
plt.clabel(CS, inline=True, fontsize=10)
plt.xlabel(r"$cost_1$", fontsize=15)
plt.ylabel(r"$cost_2$", fontsize=15)
plt.title("Spin cos-angle Distribution Function", fontsize=15)
plt.show()
