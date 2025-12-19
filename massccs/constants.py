"""
MassCCS Constants
"""

import math

# Atomic mass unit [kg]
AMU_TO_KG = 1.66053904E-27

# Mol
MOL = 6.02214078E23

# Boltzmann Constant [J/K]
BOLTZMANN_K = 1.38064E-23

# Calorie to Joule
CAL_TO_J = 4.184

# Angstrom to m
ANG_TO_M = 1.0E-10

# Femtosecond to second
FS_TO_S = 1.0E-15

# Degree to radians
DEGREE_TO_RAD = math.pi / 180.0

# Radians to degree
RAD_TO_DEGREE = 180.0 / math.pi

# Force from kcal/mol/Ang/amu to Ang/fs^2
# (1.0/48.88821291/48.88821291)
KCALMOLANGAMU_TO_ANGFS2 = 4.1840000545735656E-4

# Joule to eV
J_TO_eV = 6.242E18

# eV to kcal/mol
eV_TO_KCAL_MOL = 23.061

# Alpha to kcal/mol
ALPHA_TO_KCAL_MOL = 331.842254885

# Omega in fs^-1
OMEGA_TO_FS_INV = 2.45400505E8

# Coulomb constant in kcal/mol
KCOUL = 332.06348078020324

# amu.Ang^2/fs^2 to kcal/mol
AMU_ANG_FS2_TO_KCAL_MOL = 2390.07

# Input default parameters
NPROBE = 10000
NITER = 10
SEED = 20162104
TIMESTEP = 10.0
TEMPERATURE = 298.0
SKIN = 0.01
SHORT_CUTOFF = 12.0
INNER_SHORT_CUTOFF = 10.0
LONG_CUTOFF = 25.0
INNER_LONG_CUTOFF = 22.0
ALPHA_HE = 0.204956
ALPHA_AR = 1.6411
ALPHA_N2 = 1.710
ALPHA_CO2 = 2.911
ALPHA_co2 = 2.911

EMPTY = -1
