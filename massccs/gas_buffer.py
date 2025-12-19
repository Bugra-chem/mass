import numpy as np
import math
from .constants import *

class GasBuffer:
    def __init__(self, gas_buffer_flag):
        self.gas_buffer_flag = gas_buffer_flag

        if gas_buffer_flag == 1:
            self.natoms = 1
            self.gas_type = "He"
        elif gas_buffer_flag == 2:
            self.natoms = 3
            self.gas_type = "N2"
        elif gas_buffer_flag == 3:
            self.natoms = 3
            self.gas_type = "CO2"
        elif gas_buffer_flag == 4:
            self.natoms = 1
            self.gas_type = "Ar"
        elif gas_buffer_flag == 5:
            self.natoms = 1
            self.gas_type = "co2"
        else:
            raise ValueError(f"Unknown gas buffer flag: {gas_buffer_flag}")

        self.x = np.zeros(self.natoms)
        self.y = np.zeros(self.natoms)
        self.z = np.zeros(self.natoms)
        self.vx = np.zeros(self.natoms)
        self.vy = np.zeros(self.natoms)
        self.vz = np.zeros(self.natoms)
        self.q = np.zeros(self.natoms)
        self.m = np.zeros(self.natoms)
        self.eps = np.zeros(self.natoms)
        self.sig = np.zeros(self.natoms)
        self.atomName = [""] * self.natoms

        self.mass = 0.0
        self.d = 0.0
        self.rcm = np.zeros(3)
        self.vcm = np.zeros(3)

        # Specific for N2
        self.alpha_radial = 0.0
        self.alpha_axial = 0.0

        self.gas_properties()

    def gas_properties(self):
        if self.gas_type == "He":
            self.x[0] = 0.0
            self.y[0] = 0.0
            self.z[0] = 0.0
            self.m[0] = 4.0026
            self.eps[0] = 0.0309008
            self.sig[0] = 3.043
            self.atomName[0] = "He"
            self.mass = self.m[0]
            self.d = 0.0

        elif self.gas_type == "N2":
            # nitrogen 1
            self.x[0] = 0.0
            self.y[0] = 0.0
            self.z[0] = 0.5488
            # dummy
            self.x[1] = 0.0
            self.y[1] = 0.0
            self.z[1] = 0.0
            # nitrogen 2
            self.x[2] = 0.0
            self.y[2] = 0.0
            self.z[2] = -0.5488

            self.q[0] = -0.4825
            self.q[1] = 0.965
            self.q[2] = -0.4825
            self.m[0] = 14.007
            self.m[1] = 0.0
            self.m[2] = 14.007
            self.eps[0] = 1.0
            self.eps[1] = 0.0
            self.eps[2] = 1.0
            self.sig[0] = 0.0
            self.sig[1] = 0.0
            self.sig[2] = 0.0
            self.atomName[0] = "N_1"
            self.atomName[1] = "Dummy"
            self.atomName[2] = "N_2"
            self.mass = np.sum(self.m)
            self.d = abs(self.z[0] - self.z[2])

            # polarizability
            self.alpha_radial = 2.19609742
            self.alpha_axial = 1.51148405
            self.alpha_radial *= ALPHA_TO_KCAL_MOL
            self.alpha_axial *= ALPHA_TO_KCAL_MOL

        elif self.gas_type == "CO2":
            # oxygen 1
            self.x[0] = 0.0
            self.y[0] = 0.0
            self.z[0] = 1.149
            # carbon
            self.x[1] = 0.0
            self.y[1] = 0.0
            self.z[1] = 0.0
            # oxygen 2
            self.x[2] = 0.0
            self.y[2] = 0.0
            self.z[2] = -1.149

            self.q[0] = -0.3256
            self.q[1] = 0.6512
            self.q[2] = -0.3256
            self.m[0] = 15.999
            self.m[1] = 12.011
            self.m[2] = 15.999
            self.eps[0] = 0.159
            self.eps[1] = 0.055
            self.eps[2] = 0.159
            self.sig[0] = 3.033
            self.sig[1] = 2.757
            self.sig[2] = 3.033
            self.atomName[0] = "O1"
            self.atomName[1] = "C_"
            self.atomName[2] = "O2"
            self.mass = np.sum(self.m)
            self.d = abs(self.z[0] - self.z[2])

        elif self.gas_type == "Ar":
            self.x[0] = 0.0
            self.y[0] = 0.0
            self.z[0] = 0.0
            self.m[0] = 39.948
            self.eps[0] = 0.185
            self.sig[0] = 3.446
            self.atomName[0] = "Ar"
            self.mass = self.m[0]
            self.d = 0.0

        elif self.gas_type == "co2":
            self.x[0] = 0.0
            self.y[0] = 0.0
            self.z[0] = 0.0
            self.m[0] = 44.0098
            self.eps[0] = 0.4008
            self.sig[0] = 4.444
            self.atomName[0] = "co2"
            self.mass = self.m[0]
            self.d = 0.0
