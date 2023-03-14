"""
Testing the comoving volume calculation.
"""

import numpy as np
import matplotlib.pyplot as plt

from extrapops.cosmology import comoving_volume, diff_comoving_volume_re_z

z_range = [1e-5, 1]
z_samples = 500

zs = np.logspace(np.log10(z_range[0]), np.log10(z_range[1]), z_samples)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("V(z) in Mpc^3")
plt.plot(zs, comoving_volume(zs), "-o")
plt.loglog()
plt.subplot(122)
plt.title("dV/dz in Mpc^3")
plt.plot(zs, diff_comoving_volume_re_z(zs) * 1e-9, "-o")
plt.loglog()
plt.show()
