import numpy as np
from .constants import DEFAULT_HE_FF

class Molecule:
    def __init__(self, filename, gas_buffer_type):
        self.filename = filename
        self.gas_buffer_type = gas_buffer_type
        self.read_xyz()
        self.center_of_mass()
        self.calculate_extents()

    def read_xyz(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self.natoms = int(lines[0].strip())

        # Check format based on lines[2]
        parts = lines[2].split()
        if len(parts) == 4:
            has_charge = False
        elif len(parts) == 5:
            has_charge = True
        else:
            raise ValueError("Invalid XYZ format")

        self.mass_sum = 0.0
        self.charge_sum = 0.0

        self.x = []
        self.y = []
        self.z = []
        self.q = []
        self.eps = []
        self.sig = []
        self.m = []
        self.atom_names = []

        for i in range(2, 2 + self.natoms):
            parts = lines[i].split()
            atom_type = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            q = float(parts[4]) if has_charge else 0.0

            # Get FF params (Simplified for He)
            if self.gas_buffer_type == "He":
                params = DEFAULT_HE_FF.get(atom_type, DEFAULT_HE_FF["C"])
                m = params["m"]
                eps = params["eps"]
                sig = params["sig"]
            else:
                 m = 12.0
                 eps = 0.0
                 sig = 0.0

            self.x.append(x)
            self.y.append(y)
            self.z.append(z)
            self.q.append(q)
            self.eps.append(eps)
            self.sig.append(sig)
            self.m.append(m)
            self.atom_names.append(atom_type)

            self.mass_sum += m
            self.charge_sum += q

        self.x = np.array(self.x, dtype=np.float64)
        self.y = np.array(self.y, dtype=np.float64)
        self.z = np.array(self.z, dtype=np.float64)
        self.q = np.array(self.q, dtype=np.float64)
        self.eps = np.array(self.eps, dtype=np.float64)
        self.sig = np.array(self.sig, dtype=np.float64)
        self.m = np.array(self.m, dtype=np.float64)

    def center_of_mass(self):
        # Weighted Center of Mass
        com_x = np.sum(self.x * self.m) / self.mass_sum
        com_y = np.sum(self.y * self.m) / self.mass_sum
        com_z = np.sum(self.z * self.m) / self.mass_sum

        self.x -= com_x
        self.y -= com_y
        self.z -= com_z

        print(f"Molecule centered. Total Mass: {self.mass_sum}")

    def calculate_extents(self):
        self.maxX = np.max(np.abs(self.x))
        self.maxY = np.max(np.abs(self.y))
        self.maxZ = np.max(np.abs(self.z))
        print(f"Molecule Extents: {self.maxX}, {self.maxY}, {self.maxZ}")
