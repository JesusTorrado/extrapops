# TODO
- Implement free form R(z) (easy)
- Make an abstract function for the large-population background computation in `examples/background_MC.py`
- Save the random generator as a class attribute, so that the `lazy` generation is guaranteed at initialization to produce the same sample, regardless of calls to the RNG in the meantime.
- Make the avg SNR computation average over the spin distribution of the population.
- Make quad's faster: see https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad
- Several todo's within the code.
