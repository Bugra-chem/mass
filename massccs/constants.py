# Constants
BOLTZMANN_K = 1.38064852e-23
J_TO_eV = 6.241509126e18
eV_TO_KCAL_MOL = 23.06054801207424
AMU_TO_KG = 1.66053906660e-27
ANG_TO_M = 1.0e-10
FS_TO_S = 1.0e-15
KCALMOLANGAMU_TO_ANGFS2 = 4.184e-4
ALPHA_TO_KCAL_MOL = 332.06371

# Default FF parameters
DEFAULT_HE_FF = {
    "C": {"m": 12.011, "eps": 0.0309, "sig": 3.043},
    "N": {"m": 14.007, "eps": 0.0309, "sig": 3.043},
    "H": {"m": 1.008, "eps": 0.0150, "sig": 2.38},
    "O": {"m": 15.999, "eps": 0.0309, "sig": 3.043},
    "S": {"m": 32.06, "eps": 0.0311, "sig": 3.5},
    "P": {"m": 30.9738, "eps": 0.0311, "sig": 3.5}
}
ALPHA_HE = 0.204956
