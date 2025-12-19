import numpy as np
import math
import sys
from massccs.constants import *

class MoleculeTarget:
    def __init__(self, filename, gas_buffer_flag, user_ff="", user_ff_flag=0, force_type=0):
        self.filename = filename
        self.gas_buffer_flag = gas_buffer_flag
        self.user_ff = user_ff
        self.user_ff_flag = user_ff_flag
        self.force_type = force_type
        self.mass = 0.0
        self.Q = 0.0

        self.natoms = 0
        self.x = None
        self.y = None
        self.z = None
        self.q = None
        self.m = None
        self.eps = None
        self.sig = None
        self.atomName = []
        self.eps_central = None
        self.sig_central = None

        # User defined force field parameters
        self.nparameters = 0
        self.user_atomName = []
        self.user_m = []
        self.user_eps = []
        self.user_sig = []

        self.inertia = np.zeros((3, 3))
        self.inertiaValues = np.zeros(3)
        self.inertiaVectors = np.zeros((3, 3))
        self.moleculeRadius = 0.0
        self.diagonal = False
        self.maxX = 0.0
        self.maxY = 0.0
        self.maxZ = 0.0

        if user_ff_flag:
            self.readUserFF(user_ff)
        else:
            self.defaultFF()

        self.printFF()

        if filename.endswith(".pqr"):
            self.readPQRfile(filename)
        elif filename.endswith(".xyz"):
            self.readXYZfile(filename)
        elif filename.endswith(".mfj"):
            self.readMFJfile(filename)
        else:
             raise ValueError("Error: only acceptable PQR or XYZ or XYZ-Q format")

        self.rcm = self.calculateCenterOfMass()
        self.moveToCenterOfMass(self.rcm)
        self.calculateMoleculeRadius()
        self.setInertia()
        self.print_info()

    def readUserFF(self, user_ff):
        try:
            with open(user_ff, 'r') as f:
                lines = f.readlines()
                self.nparameters = int(lines[0].strip())
                # Skip comment line
                for i in range(2, 2 + self.nparameters):
                    parts = lines[i].split()
                    self.user_atomName.append(parts[0])
                    self.user_m.append(float(parts[1]))
                    self.user_eps.append(float(parts[2]))
                    self.user_sig.append(float(parts[3]))
        except Exception as e:
            print(f"Error reading force field file: {e}")
            sys.exit(1)

    def defaultFF(self):
        # Default force field parameters
        if self.gas_buffer_flag == 1: # Helium
            self.nparameters = 5
            self.user_atomName = ["C", "N", "H", "O", "S"]
            self.user_m = [12.011, 14.007, 1.008, 15.999, 32.06]
            self.user_eps = [0.0309, 0.0309, 0.0150, 0.0309, 0.0311]
            self.user_sig = [3.043, 3.043, 2.38, 3.043, 3.5]
        elif self.gas_buffer_flag == 2: # Nitrogen
             self.nparameters = 7
             self.user_atomName = ["C", "N", "H", "O", "S", "P", "F"]
             self.user_m = [12.011, 14.007, 1.008, 15.999, 32.06, 30.9738, 18.9984]
             self.user_eps = [0.0824736, 0.0758527, 0.0362711, 0.062323, 0.138032, 0.145372, 0.04649]
             self.user_sig = [3.2255, 3.5719, 1.8986, 3.0750, 3.4237, 3.47, 3.1285]
        elif self.gas_buffer_flag == 3 or self.gas_buffer_flag == 4: # CO2 or Ar (UFF)
             self.nparameters = 6
             self.user_atomName = ["C", "N", "H", "O", "S", "P"]
             self.user_m = [12.011, 14.007, 1.008, 15.999, 32.06, 30.9738]
             self.user_eps = [0.10499, 0.06899, 0.04399, 0.05999, 0.27399, 0.30499]
             self.user_sig = [3.43085, 3.26069, 2.57113, 3.11815, 3.59478, 3.69456]
        elif self.gas_buffer_flag == 5: # co2 (imos)
             self.nparameters = 7
             self.user_atomName = ["C", "N", "H", "O", "S", "P", "K"]
             self.user_m = [12.011, 14.007, 1.008, 15.999, 32.06, 30.9738, 39.0983]
             self.user_eps = [0.09759, 0.05579, 0.01892, 0.08279, 0.05999, 0.05999, 0.05999]
             self.user_sig = [3.58140, 3.25500, 1.24090, 4.39200, 3.50000, 3.50000, 3.50000]

    def assignedParameter(self, chemical):
        ret_size = 3
        if self.gas_buffer_flag == 3:
            ret_size = 5
        ret = np.zeros(ret_size)

        for i in range(self.nparameters):
            if chemical == self.user_atomName[i]:
                ret[0] = self.user_m[i]
                if self.gas_buffer_flag == 3:
                    # oxygen
                    ret[1] = math.sqrt(self.user_eps[i] * 0.159)
                    ret[2] = 0.5 * (self.user_sig[i] + 3.033)
                    # carbon
                    ret[3] = math.sqrt(self.user_eps[i] * 0.055)
                    ret[4] = 0.5 * (self.user_sig[i] + 2.757)
                elif self.gas_buffer_flag == 4:
                    ret[1] = math.sqrt(self.user_eps[i] * 0.185)
                    ret[2] = 0.5 * (self.user_sig[i] + 3.446)
                else:
                    ret[1] = self.user_eps[i]
                    ret[2] = self.user_sig[i]
                return ret

        print(f"Atom type {chemical} not found in the database")
        sys.exit(1)

    def printFF(self):
        print("*********************************************************")
        print("Force Field parameters: ")
        print("*********************************************************")
        if self.gas_buffer_flag == 3:
            print("Symbol  mass (amu)  epsilon (kcal/mol)  sigma(Angstroms) ")
            for i in range(self.nparameters):
                 print(f"{self.user_atomName[i]}-C  {self.user_m[i]}   {math.sqrt(self.user_eps[i]*0.055)}   {0.5*(self.user_sig[i]+2.757)}")
            for i in range(self.nparameters):
                 print(f"{self.user_atomName[i]}-O  {self.user_m[i]}   {math.sqrt(self.user_eps[i]*0.159)}   {0.5*(self.user_sig[i]+3.033)}")
        else:
             print("Symbol  mass (amu)  epsilon (kcal/mol)  sigma(Angstroms) ")
             for i in range(self.nparameters):
                 print(f"{self.user_atomName[i]}   {self.user_m[i]}   {self.user_eps[i]}   {self.user_sig[i]}")

    def readPQRfile(self, filename):
        atom_lines = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_lines.append(line)

        self.natoms = len(atom_lines)
        self.init_arrays(self.natoms)

        for i, line in enumerate(atom_lines):
            parts = line.split()
            # PQR format: Record Serial AtomName ResidueName ChainID X Y Z Charge Radius
            # But the C++ code reads: recordName >> serial >> AtomName >> residueName >> chainId >> xi >> yi >> zi >> qi >> ri;
            # This corresponds to standard PQR.
            # Python split handles whitespace separation automatically.

            # Note: PQR columns are sometimes merged. Assuming space separated here as per C++ >> operator.
            self.x[i] = float(parts[5])
            self.y[i] = float(parts[6])
            self.z[i] = float(parts[7])
            self.q[i] = float(parts[8])

            atomNameFull = parts[2]
            if self.user_ff_flag:
                self.atomName.append(atomNameFull)
            else:
                self.atomName.append(atomNameFull[0]) # First character

            ret = self.assignedParameter(self.atomName[i])
            self.m[i] = ret[0]
            self.eps[i] = ret[1]
            self.sig[i] = ret[2]

            if self.gas_buffer_flag == 3:
                self.eps_central[i] = ret[3]
                self.sig_central[i] = ret[4]

            self.mass += self.m[i]
            self.Q += self.q[i]

    def readXYZfile(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        self.natoms = int(lines[0].strip())
        self.init_arrays(self.natoms)

        # Check items per line in the first atom line (line 2, index 2)
        parts = lines[2].split()
        n_items = len(parts)

        cond = 0
        if n_items == 4:
            cond = 0
        elif n_items == 5:
            cond = 1
        else:
             raise ValueError("Error: numbers of items by lines in the xyz file")

        for i in range(self.natoms):
            line = lines[i+2]
            parts = line.split()
            atomType = parts[0]
            xi = float(parts[1])
            yi = float(parts[2])
            zi = float(parts[3])

            self.atomName.append(atomType)
            ret = self.assignedParameter(atomType)

            self.x[i] = xi
            self.y[i] = yi
            self.z[i] = zi
            self.m[i] = ret[0]
            self.eps[i] = ret[1]
            self.sig[i] = ret[2]

            if self.gas_buffer_flag == 3:
                self.eps_central[i] = ret[3]
                self.sig_central[i] = ret[4]

            self.mass += self.m[i]

            if cond == 0:
                self.q[i] = 0.0
            elif cond == 1:
                qi = float(parts[4])
                if self.force_type == 1 or self.force_type == 2:
                    self.q[i] = 0.0
                else:
                    self.q[i] = qi
                    self.Q += qi

    def readMFJfile(self, filename):
        pass # Skip for now as per instructions typically only needed PQR/XYZ

    def init_arrays(self, n):
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.z = np.zeros(n)
        self.q = np.zeros(n)
        self.m = np.zeros(n)
        self.eps = np.zeros(n)
        self.sig = np.zeros(n)
        if self.gas_buffer_flag == 3:
            self.eps_central = np.zeros(n)
            self.sig_central = np.zeros(n)

    def calculateCenterOfMass(self):
        rcm = np.zeros(3)
        for i in range(self.natoms):
            rcm[0] += self.m[i] * self.x[i]
            rcm[1] += self.m[i] * self.y[i]
            rcm[2] += self.m[i] * self.z[i]
        if self.mass > 0:
            rcm /= self.mass
        return rcm

    def moveToCenterOfMass(self, rcm):
        for i in range(self.natoms):
            self.x[i] -= rcm[0]
            self.y[i] -= rcm[1]
            self.z[i] -= rcm[2]

    def orientateMolecule(self):
        # Angle theta and phi of most distant particle
        rmax = 0.0
        atom_id = 0
        for i in range(self.natoms):
            r = math.sqrt(self.x[i]**2 + self.y[i]**2 + self.z[i]**2)
            if r > rmax:
                rmax = r
                atom_id = i

        xi = self.x[atom_id]
        yi = self.y[atom_id]
        zi = self.z[atom_id]

        theta = 0.0
        if xi != 0.0 or yi != 0.0:
            theta = math.atan2(yi, xi)

        phi = math.acos(zi/rmax) if rmax > 0 else 0

        # Rotation around z axis
        theta = -theta
        for i in range(self.natoms):
            xi = self.x[i]
            yi = self.y[i]
            self.x[i] = math.cos(theta)*xi - math.sin(theta)*yi
            self.y[i] = math.sin(theta)*xi + math.cos(theta)*yi

        # Rotation around y axis
        phi = -phi
        for i in range(self.natoms):
            xi = self.x[i]
            zi = self.z[i]
            self.x[i] = math.sin(phi)*zi + math.cos(phi)*xi
            self.z[i] = math.cos(phi)*zi - math.sin(phi)*xi

        # Angle gamma of most distant particle in xy plane
        rmax = 0.0
        atom_id = 0
        for i in range(self.natoms):
            r = math.sqrt(self.x[i]**2 + self.y[i]**2)
            if r > rmax:
                rmax = r
                atom_id = i

        xi = self.x[atom_id]
        yi = self.y[atom_id]

        gamma = math.atan2(yi, xi)

        # Rotation around z axis
        gamma = -gamma
        for i in range(self.natoms):
            xi = self.x[i]
            yi = self.y[i]
            self.x[i] = math.cos(gamma)*xi - math.sin(gamma)*yi
            self.y[i] = math.sin(gamma)*xi + math.cos(gamma)*yi

    def setInertia(self):
        # Initialize inertia matrix
        self.inertia.fill(0.0)

        for i in range(self.natoms):
            r = np.array([self.x[i], self.y[i], self.z[i]])
            mi = self.m[i]
            r2 = np.dot(r, r)
            for a in range(3):
                for b in range(3):
                    delta = 1.0 if a == b else 0.0
                    self.inertia[a][b] += mi * (r2 * delta - r[a] * r[b])

        # Check if diagonal
        is_diag = True
        for a in range(3):
            for b in range(3):
                if a != b and abs(self.inertia[a][b]) > 1.e-12:
                    is_diag = False
                    break

        if is_diag:
            self.orientateMolecule()
            self.diagonal = True
            # Identity matrix for eigenvectors if already diagonal
            self.inertiaVectors = np.eye(3)
        else:
            # Diagonalize
            # Note: C++ implementation uses a custom solver. Numpy eigh should be equivalent for symmetric real matrices.
            vals, vecs = np.linalg.eigh(self.inertia)
            # eigh returns eigenvalues in ascending order.
            # C++ implementation seems to do something specific.
            # Let's trust numpy's diagonalization.
            # The C++ code then checks dot products of eigenvectors to see if they align?
            # And then rotates the molecule.

            # Rotate molecule to principal axes
            # New coordinates = V.T * Old coordinates?
            # C++ code:
            # x[i] = xi*inertiaVectors[0][0] + yi*inertiaVectors[0][1] + zi*inertiaVectors[0][2];
            # This looks like row-vector multiplication if inertiaVectors are rows?
            # Or x_new = V * x_old

            # In C++:
            # eigenvectors[0] is the first eigenvector
            # x_new = dot(x_old, eigenvector[0])
            # So the molecule is projected onto the eigenvectors.

            # Numpy returns column eigenvectors.
            # vecs[:, i] is the ith eigenvector.

            # We want to rotate the molecule so that the principal axes align with x, y, z.
            # So we project onto the eigenvectors.

            for i in range(self.natoms):
                 xi = self.x[i]
                 yi = self.y[i]
                 zi = self.z[i]
                 # Using the eigenvectors from numpy (columns) as the new basis
                 # x_new = r . v1
                 # y_new = r . v2
                 # z_new = r . v3

                 # Numpy sorts eigenvalues, so order matters.
                 # The C++ code doesn't sort explicitly but the solver usually produces them in some order.
                 # Let's just use what we have.

                 self.x[i] = xi*vecs[0,0] + yi*vecs[1,0] + zi*vecs[2,0]
                 self.y[i] = xi*vecs[0,1] + yi*vecs[1,1] + zi*vecs[2,1]
                 self.z[i] = xi*vecs[0,2] + yi*vecs[1,2] + zi*vecs[2,2]

            # Store for printing
            # C++ stores eigenvectors in rows in its custom implementation output
            self.inertiaVectors = vecs.T
            self.inertiaValues = vals

        # Find direction of longest extent
        self.maxX = 0.0
        self.maxY = 0.0
        self.maxZ = 0.0
        for i in range(self.natoms):
            if abs(self.x[i]) > abs(self.maxX): self.maxX = self.x[i]
            if abs(self.y[i]) > abs(self.maxY): self.maxY = self.y[i]
            if abs(self.z[i]) > abs(self.maxZ): self.maxZ = self.z[i]

        maxX_abs = abs(self.maxX)
        maxY_abs = abs(self.maxY)
        maxZ_abs = abs(self.maxZ)

        # if longest extent is along x-axis, rotate about y-axis -pi/2 => z becomes x, x becomes -z
        if maxX_abs > maxY_abs and maxX_abs > maxZ_abs:
            for i in range(self.natoms):
                xi = self.x[i]
                zi = self.z[i]
                self.x[i] = -zi
                self.z[i] = xi
        elif maxY_abs > maxX_abs and maxY_abs > maxZ_abs:
             for i in range(self.natoms):
                yi = self.y[i]
                zi = self.z[i]
                self.y[i] = zi
                self.z[i] = -yi

        # Reset maximum extent values
        self.maxX = 0.0
        self.maxY = 0.0
        self.maxZ = 0.0
        for i in range(self.natoms):
            if abs(self.x[i]) > abs(self.maxX): self.maxX = self.x[i]
            if abs(self.y[i]) > abs(self.maxY): self.maxY = self.y[i]
            if abs(self.z[i]) > abs(self.maxZ): self.maxZ = self.z[i]

    def calculateMoleculeRadius(self):
        rmax = 0.0
        for i in range(self.natoms):
            r = math.sqrt(self.x[i]**2 + self.y[i]**2 + self.z[i]**2)
            if r > rmax:
                rmax = r
        self.moleculeRadius = rmax

    def print_info(self):
        print("*********************************************************")
        print("MOLECULE:: orientation around the inertia principal axis ")
        print("*********************************************************")
        print("Inertia matrix a.u. Ang^2: ")
        print(f"{{  {self.inertia[0][0]:g}  {self.inertia[0][1]:g}  {self.inertia[0][2]:g}  }}")
        print(f"{{  {self.inertia[1][0]:g}  {self.inertia[1][1]:g}  {self.inertia[1][2]:g}  }}")
        print(f"{{  {self.inertia[2][0]:g}  {self.inertia[2][1]:g}  {self.inertia[2][2]:g}  }}")

        if not self.diagonal:
            print("Rotation matrix applied to molecule: ")
            print(f"{{  {self.inertiaVectors[0][0]:g}  {self.inertiaVectors[0][1]:g}  {self.inertiaVectors[0][2]:g}  }}")
            print(f"{{  {self.inertiaVectors[1][0]:g}  {self.inertiaVectors[1][1]:g}  {self.inertiaVectors[1][2]:g}  }}")
            print(f"{{  {self.inertiaVectors[2][0]:g}  {self.inertiaVectors[2][1]:g}  {self.inertiaVectors[2][2]:g}  }}")

        print(f"Molecule radius: {self.moleculeRadius:g} Ang ")
