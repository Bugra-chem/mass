import numpy as np
import math

class MoleculeTarget:
    def __init__(self, filename, gas_buffer_flag=2):
        self.filename = filename
        self.gas_buffer_flag = gas_buffer_flag
        self.natoms = 0
        self.atomName = []
        self.x = []
        self.y = []
        self.z = []
        self.q = []
        self.m = []
        self.eps = []
        self.sig = []
        self.mass = 0.0
        self.Q = 0.0
        self.rcm = np.zeros(3)
        self.maxX = 0.0
        self.maxY = 0.0
        self.maxZ = 0.0
        self.moleculeRadius = 0.0

        self.readXYZfile(filename)
        self.calculateCenterOfMass()
        self.moveToCenterOfMass()
        self.calculateMoleculeRadius()
        self.setInertia()

    def readXYZfile(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        self.natoms = int(lines[0].strip())
        # lines[1] is comment

        for i in range(2, 2 + self.natoms):
            parts = lines[i].split()
            atomType = parts[0]
            xi = float(parts[1])
            yi = float(parts[2])
            zi = float(parts[3])
            qi = 0.0
            if len(parts) > 4:
                qi = float(parts[4])

            self.atomName.append(atomType)
            self.x.append(xi)
            self.y.append(yi)
            self.z.append(zi)
            self.q.append(qi)

            params = self.assignedParameter(atomType)
            mi = params[0]
            self.m.append(mi)
            self.eps.append(params[1])
            self.sig.append(params[2])

            self.mass += mi
            self.Q += qi

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        self.q = np.array(self.q)
        self.m = np.array(self.m)
        self.eps = np.array(self.eps)
        self.sig = np.array(self.sig)

    def assignedParameter(self, chemical):
        # Default force field parameters (UFF/Standard)
        # Copied from C++ defaultFF
        # N2 flag is 2.

        # Only implementing for N2 gas buffer (flag 2) as per instructions
        if self.gas_buffer_flag == 2:
            if chemical == "C":
                return [12.011, 0.0824736, 3.2255]
            elif chemical == "N":
                return [14.007, 0.0758527, 3.5719]
            elif chemical == "H":
                return [1.008, 0.0362711, 1.8986]
            elif chemical == "O":
                return [15.999, 0.062323, 3.0750]
            elif chemical == "S":
                return [32.06, 0.138032, 3.4237]
            elif chemical == "P":
                return [30.9738, 0.145372, 3.47]
            elif chemical == "F":
                return [18.9984, 0.04649, 3.1285]

        # Fallback for other atoms if needed, or error
        raise ValueError(f"Atom type {chemical} not found in database for gas flag {self.gas_buffer_flag}")

    def calculateCenterOfMass(self):
        self.rcm = np.zeros(3)
        for i in range(self.natoms):
            self.rcm[0] += self.m[i] * self.x[i]
            self.rcm[1] += self.m[i] * self.y[i]
            self.rcm[2] += self.m[i] * self.z[i]
        self.rcm /= self.mass

    def moveToCenterOfMass(self):
        self.x -= self.rcm[0]
        self.y -= self.rcm[1]
        self.z -= self.rcm[2]
        self.rcm = np.zeros(3)

    def calculateMoleculeRadius(self):
        r2 = self.x**2 + self.y**2 + self.z**2
        self.moleculeRadius = np.sqrt(np.max(r2))

    def setInertia(self):
        # Calculate inertia tensor
        inertia = np.zeros((3, 3))
        for i in range(self.natoms):
            r = np.array([self.x[i], self.y[i], self.z[i]])
            r2 = np.dot(r, r)
            mi = self.m[i]
            for a in range(3):
                for b in range(3):
                    delta = 1.0 if a == b else 0.0
                    inertia[a][b] += mi * (r2 * delta - r[a] * r[b])

        # Diagonalize
        evals, evecs = np.linalg.eigh(inertia)

        # Sort eigenvalues/vectors (numpy eigh already sorts them ascending)
        # C++ implementation seems to sort them differently or just use them
        # Let's align principal axes with Cartesian axes

        # Transform coordinates
        # evecs columns are the eigenvectors
        rotation_matrix = evecs.T

        new_coords = np.zeros((self.natoms, 3))
        for i in range(self.natoms):
            r = np.array([self.x[i], self.y[i], self.z[i]])
            new_coords[i] = np.dot(rotation_matrix, r)

        self.x = new_coords[:, 0]
        self.y = new_coords[:, 1]
        self.z = new_coords[:, 2]

        # Find direction of longest extent
        maxX = np.max(np.abs(self.x))
        maxY = np.max(np.abs(self.y))
        maxZ = np.max(np.abs(self.z))

        # Align longest extent along Z (C++ code does something specific here)
        # C++: if maxX > maxY and maxX > maxZ: rotate around Y by -pi/2
        #      else if maxY > maxX and maxY > maxZ: rotate around X by pi/2 (effectively)

        if maxX > maxY and maxX > maxZ:
            # Rotate -pi/2 around Y: x' = -z, z' = x
            new_x = -self.z
            new_z = self.x
            self.x = new_x
            self.z = new_z
        elif maxY > maxX and maxY > maxZ:
            # Rotate around X to bring Y to Z?
            # C++: y' = z, z' = -y
            new_y = self.z
            new_z = -self.y
            self.y = new_y
            self.z = new_z

        self.maxX = np.max(np.abs(self.x))
        self.maxY = np.max(np.abs(self.y))
        self.maxZ = np.max(np.abs(self.z))
