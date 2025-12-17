import numpy as np
from .constants import *

def calculate_reduced_mass(target_mass, gas_type):
    if gas_type == "He":
        gas_mass = 4.0026
    elif gas_type == "N2":
        gas_mass = 28.0134
    elif gas_type == "Ar":
        gas_mass = 39.948
    elif gas_type.upper() == "CO2":
        gas_mass = 44.01
    else:
        gas_mass = 4.0

    return (target_mass * gas_mass) / (target_mass + gas_mass)

def calculate_ellipsoid(molecule, lj_cutoff, skin):
    radius = 2.0 * lj_cutoff
    a = molecule.maxX + radius + skin
    b = molecule.maxY + radius + skin
    c = molecule.maxZ + radius + skin
    bmax = max(a, max(b, c))
    return a, b, c, bmax

def velDistr(v, m, temperature):
    # m in kg, v in m/s
    return (v**5) * ((m / (2.0 * BOLTZMANN_K * temperature))**3) * np.exp(-(m * v**2) / (2.0 * BOLTZMANN_K * temperature))

def velGenerator_host(m, temperature, sd):
    # m is in AMU, must convert to kg for distribution
    m_kg = m * AMU_TO_KG

    v_angs_fs = 0.0
    dv_angs_fs = 1.0E-5

    sumProbability = 0.0

    # We loop until sumProbability > sd (which is 0..1)?
    # Wait, the C++ code accumulates `velDistr`.
    # Does velDistr integral sum to 1?
    # Integral x^5 exp(-x^2) dx = 1/2 * Gamma(3) = 1.
    # The normalization constant might be off or correct.
    # We should trust C++ logic: accumulate until > random.

    # Safety limit to avoid infinite loop
    MAX_STEPS = 200000
    steps = 0

    # Optimization: Pre-calculate constants
    # But for a single value we iterate.
    # This is extremely slow in Python if done step by step.
    # We can pre-compute a table or use inverse transform sampling if distribution is known.
    # The C++ code is inefficient but simple.

    # Let's try to emulate C++ logic but vectorized or optimized if possible.
    # Or just run it brutally for verification.

    # For CPU implementation, we might need to be smarter.
    # The distribution is f(v) ~ v^5 exp(-av^2).
    # If we integrate f(v) dv, we get CDF.
    # We want v such that CDF(v) = sd.
    # Since we can't analytically invert easily (it involves Gamma functions),
    # we can numerically solve it or use rejection sampling.

    # But to match C++ exactly:

    current_v_ms = 0.0
    # In C++, v is passed to velDistr in Ang/fs, converted to m/s inside.
    # Then returns val.

    # Let's write Python version of velDistr matching C++ exactly
    def vel_distr_cpp(v_angs, m_amu, T):
        m_kg = m_amu * AMU_TO_KG
        v_ms = v_angs * ANG_TO_M / FS_TO_S
        val = (v_ms**5) * ((m_kg / (2.0 * BOLTZMANN_K * T))**3) * np.exp(-(m_kg * v_ms**2) / (2.0 * BOLTZMANN_K * T))
        return val

    # Since this is slow, maybe pre-build a CDF table?
    # No, let's just do it for now.

    # Actually, for the CPU engine (batch processing), we can vectorise this.
    # Generate array of `v` values, compute cumsum (CDF), and interpolate.

    v_grid = np.arange(0, 0.1, dv_angs_fs) # 0 to 0.1 A/fs (10000 m/s) is huge enough
    # He at 300K: v_rms = sqrt(3kT/m) ~ 1300 m/s ~ 0.013 A/fs.
    # So 0.1 is plenty.

    pdf = vel_distr_cpp(v_grid, m, temperature)
    cdf = np.cumsum(pdf) # implicitly * 1 step (C++ didn't multiply by dv? wait)

    # C++: sumProbability += velDistr(v...);
    # It does NOT multiply by dv. So it sums raw PDF values.
    # So we should just cumsum.

    # Normalize?
    # In C++, sd is random(0,1).
    # If sumProbability (CDF) doesn't reach 1, we have a problem.
    # If it reaches >1, we stop.
    # Let's see max value of CDF.

    cdf_max = cdf[-1]
    # If C++ code assumes CDF goes to infinity or some value?
    # If sd is uniform(0,1), and max CDF is say 1000, we pick very small v.
    # If max CDF is 0.001, we loop forever.
    # We should normalize CDF to match sd range if physics implies it.
    # But C++ code implies direct comparison.

    # If the user says "result is wrong", maybe the velocity generator is broken in Python because I assumed it works like standard inversion sampling but C++ code does something specific.
    # The C++ code:
    # sumProbability = velDistr(0, ...)
    # while (sumProbability < sd) { v += dv; sumProbability += velDistr(v...); }
    # This implies sd should be drawn from range [0, total_integral].
    # But sd is passed as `mt->getRandomNumber()` which usually is [0,1].

    # Let's check `velDistr` magnitude.
    # m ~ 6e-27. T=300. 2kT ~ 8e-21. m/2kT ~ 0.7e-6. (m/2kT)^3 ~ 3e-19.
    # v ~ 1000. v^5 ~ 1e15.
    # exp(...) ~ 1.
    # Result ~ 1e15 * 3e-19 ~ 3e-4.
    # Summing 3e-4 for many steps...
    # If v range is 0.05 A/fs -> 5000 steps.
    # 5000 * 3e-4 ~ 1.5.
    # So the sum DOES reach order of 1.
    # So comparing to uniform(0,1) is correct!

    # Vectorized inverse transform sampling:
    return np.interp(sd, cdf, v_grid)
