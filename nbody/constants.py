"""Physical constants and unit conventions.

Units: lengths in astronomical units (AU), masses in solar masses (M_sun),
times in years. With these units Kepler's third law for a massless body
orbiting one solar mass at 1 AU gives a period of exactly one year, so

    G = 4 * pi**2   [AU^3 / (M_sun * yr^2)].

That "year" is the Gaussian year, 2*pi/k days, where k is the Gaussian
gravitational constant. Source for k: IAU (1976) System of Astronomical
Constants, k = 0.01720209895 AU^(3/2) M_sun^(-1/2) day^(-1) (exact by
definition). We need it to convert JPL Horizons data, which is in AU and days.
"""

import math

G: float = 4.0 * math.pi**2
"""Gravitational constant in AU^3 / (M_sun yr^2)."""

GAUSSIAN_K: float = 0.01720209895
"""Gaussian gravitational constant, IAU 1976 (AU^1.5 M_sun^-0.5 day^-1)."""

DAYS_PER_YEAR: float = 2.0 * math.pi / GAUSSIAN_K
"""Length of our time unit in days: the Gaussian year, 2*pi/k = 365.2568983... days."""
