import numpy as np
from scipy import integrate, interpolate


# From https://mathworld.wolfram.com/RandomNumber.html
def power_law_generate(N, power, x_min, x_max):
    pow_loc, pow_scale = x_min**(1 + power), x_max**(1 + power) - x_min**(1 + power)
    return (pow_loc + pow_scale * np.random.random(N))**(1 / (1 + power))


def invCDFinterp(x, pdf_func, pdf_args=None, splrep_kwargs=None):
    """
    Prepares an interpolator for invCDF sampling.

    ``x`` samples are assumed sorted.

    Integrates the pdf ``pdf_func`` using quad. Use ``pdf_args`` to pass arguments to it
    at integration.

    Uses ``scipy.interpolate.splrep`` to create an interpolator. Use ``splrep_kwargs`` to
    pass keyword arguments to it, e.g. ``{"k": 1}``.

    Returns a tuple of the the CDF samples and the interpolator.
    """
    quad_kwargs = {"args": pdf_args} if pdf_args else {}
    CDF = np.array(
        [integrate.quad(pdf_func, x[0], x_i, **quad_kwargs)[0] for x_i in x]
    )
    # Sometimes (very rarely) not sorted due to numerical noise. Simply delete bad entry
    for _ in range(len(CDF) - 1):
        i_unsorted_left = np.argwhere(np.diff(CDF) < 0)
        if i_unsorted_left.shape[0]:  # unsorted
            CDF = np.delete(CDF, i_unsorted_left, axis=0)
            x = np.delete(x, i_unsorted_left, axis=0)
        else:
            break
    else:
        raise ValueError(
            "Could not produce a sorted sample of the CDF. "
            "Maybe the pdf passed is a stochastic function?"
        )
    # Normalise to [0, 1]
    CDF = (CDF - min(CDF)) / (max(CDF) - min(CDF))
    return (CDF, interpolate.splrep(CDF, x, **(splrep_kwargs or {})))
