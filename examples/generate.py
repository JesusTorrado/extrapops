"""
Generates a population with given parameters.
"""

import os
from pprint import pprint
import matplotlib.pyplot as plt
from yaml import safe_load

from conftest import _output_folder, ensure_output_folder
from extrapops.population import Population


# Use GWTC-3 median
filename = "gwtc3_median.yaml"
# Use fiducial fixed-point for the Background paper
filename = "N18.yaml"

with open(filename, "r") as f:
    population_params = safe_load("\n".join(f.readlines()))

print("Population parameters:")
pprint(population_params)

# Reducing z and tc range for tests
population_params["redshift"]["z_range"][1] = 1
population_params["redshift"]["T_yr"] = 1000

# Generate population and do some plots
print("Generating population...")
pop = Population(**population_params)
print(f"Population generated with {len(pop)} inspirals!")

pop.plot("CoalTime", "Redshift")
pop.plot("Mass1", "Mass2")
plt.show(block=False)

# Save+load test
file_name = "testpop"
ensure_output_folder()
out_path = os.path.join(_output_folder, file_name)
print(f"Saving population at {out_path}...")
pop.save(out_path, force=True, LISA=True)
print("Population saved!")

# Check that it is loaded correctly, including frame transform
print(f"Loading population from {out_path}...")
pop2 = Population(load=out_path)
print("Population loaded!")
plt.figure()
pop.plot("CoalTime", "Redshift")
plt.gca().set_title("(after save + load)")
pop.plot("Mass1", "Mass2")
plt.gca().set_title("(after save + load)")
plt.show(block=True)
