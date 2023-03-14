"""
Generate samples from the event rate as a function of redshift.
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm

import extrapops.constants as const
from extrapops.redshifts import (
    sample_z,
    event_rate_persec,
    avg_n_events,
    _invCDF_cache_params,
)

T_yr = 10000

start = time()
z_dist_kwargs = {"T_yr": T_yr}
z_sample = sample_z(**z_dist_kwargs)
end = time()

# print times with and without cache
# import after caching

print(
    "Generated %d events (of avg %d) in %f sec!"
    % (len(z_sample), avg_n_events(**z_dist_kwargs), end - start)
)

print("Params:", _invCDF_cache_params)

z_range = [1e-5, 1]
z_samples = 500
zmin, zmax = z_range
zs = np.logspace(np.log10(z_range[0]), np.log10(z_range[1]), z_samples)
pdf_z = lambda z: T_yr * const.yr_s * event_rate_persec(z)

pdf_color = "tab:orange"
events_color = "tab:blue"

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Dist and sample")
plt.hist(z_sample, density=True, bins=500, color=events_color)
plt.plot(zs, pdf_z(zs) / len(z_sample), color=pdf_color)
# plt.xlim(zmin, zmax)

plt.subplot(132)
plt.title("Zoom low z (noisier events: important!)")
plt.hist(z_sample, density=True, bins=500, color=events_color)
plt.plot(zs, pdf_z(zs) / len(z_sample), color=pdf_color)
plt.axvline(zmin, ls="--", color="0.5")
plt.loglog()
plt.xlim(0.5 * zmin, 0.1)

plt.subplot(133)
plt.title("Zoom high z")
plt.hist(z_sample, density=True, bins=500, color=events_color)
plt.plot(zs, pdf_z(zs) / len(z_sample), color=pdf_color)
plt.axvline(zmin, ls="--", color="0.5")
plt.axvline(zmax, ls="--", color="0.5")
plt.semilogx()
plt.xlim(0.9 * zmax, zmax * 1.05)
plt.show()

# Now testing low-z tail with high precision: very few but possibly detectable!

print("Testing probability of low-z events. This may take a while...")

ndraws_below = 10
max_low_z_test = 0.001
max_low_z_plot = 2 * max_low_z_test  # larger that max_low_z_test

print("Requested %d events below %g" % (ndraws_below, max_low_z_test))

get_ndraws_below_perpop = lambda z: integrate.quad(pdf_z, z_range[0], z)[0]

ndraws_below_perpop = get_ndraws_below_perpop(max_low_z_test)
npops = int(np.ceil(ndraws_below / ndraws_below_perpop))

print(
    "Expected %g events per population --> ~%d populations needed."
    % (ndraws_below_perpop, npops)
)

low_z_sample = []

for _ in tqdm(range(npops)):
    z_sample = sample_z(T_yr=T_yr)
    low_z_sample += list(z_sample[np.where(z_sample < max_low_z_plot)])
    n_so_far = len(np.where(np.array(low_z_sample) <= max_low_z_test)[0])
    print("So far %d / %d" % (n_so_far, ndraws_below))


zs = np.logspace(np.log10(z_range[0]), np.log10(max_low_z_plot), z_samples)

plt.figure()
plt.title("Zoom low z (noisier events: important!)")
plt.hist(low_z_sample, density=True, bins=500, color=events_color)
plt.plot(zs, pdf_z(zs), color=pdf_color)
plt.axvline(zmin, ls="--", color="0.5")
plt.axvline(max_low_z_test, ls="--", color="0.5")
plt.axvline(max_low_z_plot, ls="--", color="0.5")
plt.loglog()
plt.show()



# probar este y todos los tests
# borrar la parte de mauro!!!!!!
# y que solo se guarden plots en output/ y ponerlo en gitignore, y crearlo

