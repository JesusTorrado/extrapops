"""
A script to compute a Monte Carlo background for a population.

May need to be used instead of the relevant method in the Population class if the
population size is very big (events are generated in chunks and then dropped from memory).
"""

import os
import sys
from itertools import chain

import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from yaml import safe_load
import matplotlib.pyplot as plt
import matplotlib as mpl

from extrapops.redshifts import draw_Nsamples, sample_z, _default_z_perdecade
from extrapops.mass import chirp_mass, sample_m1_m2
from extrapops.cosmology import luminosity_distance_to_redshift
from extrapops.background import (
    char_strain_sq_single_at_f1Hz,
    char_strain_sq_numerical,
    char_strain_squared_to_Omegah2,
)
import extrapops.constants as const
from conftest import _output_folder, ensure_output_folder

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()
MPI_IS_MAIN = not bool(MPI_RANK)

# Matpotlib settings
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=18)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=18)  # fontsize of the y tick labels
plt.rc("legend", fontsize=16)  # fontsize of the legend labels


# Scientific formatter; takes str, does not return $$
def scifmt(numstr):
    if not isinstance(numstr, str):
        numstr = f"{numstr:g}"
    if "e" in numstr:
        return f"{numstr.split('e')[0]} \\cdot 10^{{ {numstr.split('e')[1]} }}"
    else:
        return numstr


# TODO: maybe generalise and put somewhere in the main tree?
def single_pop_background(chunk_size=1e6, population_params=None):
    """
    Computes the MC-summed background for a single population, without retaining the
    events in memory.

    Returns the characteristic strain squared (unitless) at f=1Hz.
    """
    Nsamples = draw_Nsamples(
        T_yr=population_params["redshift"]["T_yr"],
        merger_rate_model=population_params["redshift"]["merger_rate_model"],
        merger_rate_params=population_params["redshift"]["merger_rate_params"],
        z_range=population_params["redshift"]["z_range"],
        z_perdecade=population_params["redshift"].get(
            "z_perdecade", _default_z_perdecade),
        cosmo_params=population_params["cosmo"],
    )
    chunk_sizes = int(np.floor(Nsamples / chunk_size)) * [int(chunk_size)]
    chunk_sizes += [int(Nsamples % chunk_size)]
    char_strain_sq_1Hz = 0
    for i, size_i in enumerate(chunk_sizes):
        # Generate the poplation chunk and sum the background
        z_sample = sample_z(
            size_i,
            cosmo_params=population_params["cosmo"],
            **population_params["redshift"],
        )
        M_sample = chirp_mass(*sample_m1_m2(size_i, **population_params["mass"]))
        lum_distances = (
            luminosity_distance_to_redshift(
                z_sample,
                z_range=population_params["redshift"]["z_range"],
                z_perdecade=population_params["redshift"].get(
                    "z_perdecade", _default_z_perdecade),
                **population_params["cosmo"],
            )
            * 1e-3
        )
        char_strain_sq_1Hz += np.sum(
            char_strain_sq_single_at_f1Hz(
                z_sample,
                M_sample * const.Msun_kg,
                lum_distances * const.Gpc_m / (1 + z_sample),
                T_gen=population_params["redshift"]["T_yr"] * const.yr_s,
            )
        )
    return char_strain_sq_1Hz


def generate_pops(n_pops, file_name, population_params, chunk_size=1e6, reset=False):
    if reset or not os.path.exists(file_name):
        if MPI_IS_MAIN:
            print("First computing num. integral of the analytic background...", end="")
            sys.stdout.flush()
            num = char_strain_sq_numerical(
                cosmo_params=population_params["cosmo"],
                redshift_params=population_params["redshift"],
                mass_params=population_params["mass"],
            )
            print("Done!")
            with open(file_name, "w") as f:
                f.write(
                    "# Characteristic strain squared (unitless) at f=1Hz\n"
                    "# (1st row is the numerical integral of the analytic formula.)\n"
                )
                f.write(f"{num:g}\n")
        MPI_COMM.barrier()
    char_strains_sq_at_1Hz = []
    iterator_wrapper = tqdm if MPI_IS_MAIN else lambda x: x
    for _ in iterator_wrapper(range(n_pops)):
        char_strains_sq_at_1Hz.append(
            single_pop_background(
                chunk_size=chunk_size, population_params=population_params
            )
        )
    MPI_COMM.barrier()
    all_char_strains_sq_at_1Hz = MPI_COMM.gather(char_strains_sq_at_1Hz)
    if MPI_IS_MAIN:
        all_char_strains_sq_at_1Hz = list(chain(*all_char_strains_sq_at_1Hz))
        print("Generated %d populations." % len(all_char_strains_sq_at_1Hz))
        with open(file_name, "a") as f:
            f.write("\n".join([f"{val:g}" for val in all_char_strains_sq_at_1Hz]) + "\n")


def plot_pops(file_name, f=1, omh2=True):
    if not MPI_IS_MAIN:
        return
    if not os.path.exists(file_name):
        raise ValueError("Realisations file not found: %r" % file_name)
    strains_at_1Hz = np.loadtxt(file_name)
    strains_at_f = np.array(strains_at_1Hz) * f ** (-4 / 3)
    num, strains_at_f = strains_at_f[0], strains_at_f[1:]
    # Rounding of #realisations for published figure -- round to hundreds
    strains_at_f = strains_at_f[:100 * (len(strains_at_f) // 100)]
    avg, std = np.mean(strains_at_f), np.std(strains_at_f)
    med = np.percentile(strains_at_f, 50)
    iqr = np.percentile(strains_at_f, 75) - np.percentile(strains_at_f, 25)
    nbins = 150
    hist_color = "tab:blue"
    f_pivot_fmt = scifmt(f"{f:g}")
    if omh2:
        to_plot = char_strain_squared_to_Omegah2(f, strains_at_f)
        num = char_strain_squared_to_Omegah2(f, num)
        avg = char_strain_squared_to_Omegah2(f, avg)
        std = char_strain_squared_to_Omegah2(f, std)
        med = char_strain_squared_to_Omegah2(f, med)
        iqr = char_strain_squared_to_Omegah2(f, iqr)
        quantity_str = f"h^2\\Omega_\\mathrm{{GW}}(f={f_pivot_fmt} \\mathrm{{Hz}})"
    else:
        to_plot = strains_at_f
        quantity_str = f"h_c^2(f={f:g}) \\mathrm{{Hz}} (unitless)"
    print(quantity_str + " = %g +/- %g (%g %%), IQR/2=%g (%g %%) (#=%d)" %
          (avg, std, 100 * std / avg, iqr / 2, 100 * iqr / 2 / avg, len(strains_at_f)))
    plt.figure(figsize=(8, 5))
    vals, bin_edges, _ = plt.hist(to_plot, bins=nbins, density=True, color=hist_color,
                                  label="Frequency of MC background amplitude")
    plt.xlim(avg - 1.5 * std, avg + 3.5 * std)  # asymmetric dist
    plt.xlabel(f"${quantity_str}$", fontsize=16)
    plt.ylabel('Probability density', fontsize=16)
    lw = 1.5
    plt.axvline(avg, lw=lw, ls="--", c="k",
                label=(f"Avg. {len(strains_at_f)} "
                       f"MC realisations: ${scifmt(f'{avg:.4g}')}$"))
    # plt.axvline(med, lw=lw, ls=":", c="k",
    #             label=(f"Median {len(strains_at_f)} "
    #                    f"MC realisations: ${scifmt(f'{med:.4g}')}$"))
    plt.axvline(num, lw=lw, c="k",
                label=f"Numerical integration: ${scifmt(f'{num:.4g}')}$")
    others = {
        # "legend_label": value,
    }
    lstyles = ["-", "--", "-.", ":"]
    for i, (label, val) in enumerate(others.items()):
        plt.axvline(val, label=label + " (%g)" % val, c="k", ls=lstyles[i])
    # Gridlines configuration
    plt.grid(which='both', color='grey', linestyle='-', linewidth=0.3, alpha=.8)
    plt.legend()
    plt.tight_layout()
    plot_file_name = os.path.join(_output_folder, "realisations.pdf")
    plt.savefig(plot_file_name)
    print(f"Plot saved at '{plot_file_name}'")
    plt.show()


if __name__ == "__main__":
    params_file_name = "N18.yaml"
    output_file_name = "realisations_N18_MadauFragos16.dat"
    reset = False  # if False, append; if True, overwrite.
    ensure_output_folder()
    full_file_name = os.path.join(_output_folder, output_file_name)
    f_pivot = 3e-3  # Hz
    with open(params_file_name, "r") as params_file:
        population_params = safe_load(params_file)
    try:
        sys.argv[1] = int(sys.argv[1])
    except (IndexError, ValueError):
        pass
    if len(sys.argv) == 2 and str(sys.argv[1]).lower() == "plot":
        plot_pops(full_file_name, f_pivot)
    elif len(sys.argv) == 2 and isinstance(sys.argv[1], int):
        generate_pops(
            int(sys.argv[1]),
            full_file_name,
            population_params,
            chunk_size=1e7,  # approx 1GB per 1e7
            reset=reset,
        )
    else:
        raise ValueError(
            f"Pass the number of events to generate, or 'plot'. Passed {sys.argv[1:]}"
        )
