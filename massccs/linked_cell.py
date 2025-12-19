import numpy as np
import math
import sys

class LinkedCell:
    def __init__(self, moleculeTarget, a, b, c, lj_cutoff, skin, long_range_flag, long_range_cutoff, coul_cutoff, gas_buffer_flag):
        self.moleculeTarget = moleculeTarget
        self.a = a
        self.b = b
        self.c = c
        self.lj_cutoff = lj_cutoff
        self.skin = skin
        self.long_range_flag = long_range_flag
        self.long_range_cutoff = long_range_cutoff
        self.coul_cutoff = coul_cutoff
        self.gas_buffer_flag = gas_buffer_flag
        self.next_neighbor = 0

        self.lx = 2.0 * a
        self.ly = 2.0 * b
        self.lz = 2.0 * c

        self.Nx = 0
        self.Ny = 0
        self.Nz = 0
        self.Ncells = 0

        self.atoms_inside_cell = None
        self.head_atom_cell = None
        self.atoms_ids = []

        self.neighbors1_cells = None
        self.neighbors1_cells_ids = []

        self.neighbors2_cells = None
        self.neighbors2_cells_ids = []

        self.corner = np.zeros(3)

        self.calculateNumberOfCells()
        self.calculateAtomsInsideOfCell()
        self.sortingAtoms()
        self.calculateCellsNeighbors()
        self.print_info()

    def calculateNumberOfCells(self):
        # Number of cells for each axis
        self.Nx = int(math.ceil(self.lx / (self.lj_cutoff + self.skin)))
        self.Ny = int(math.ceil(self.ly / (self.lj_cutoff + self.skin)))
        self.Nz = int(math.ceil(self.lz / (self.lj_cutoff + self.skin)))

        # Adjust box simulation domain
        self.lx = float(self.Nx) * (self.lj_cutoff + self.skin)
        self.ly = float(self.Ny) * (self.lj_cutoff + self.skin)
        self.lz = float(self.Nz) * (self.lj_cutoff + self.skin)

        self.Ncells = self.Nx * self.Ny * self.Nz

        self.atoms_inside_cell = np.zeros(self.Ncells, dtype=int)
        self.head_atom_cell = np.zeros(self.Ncells, dtype=int)

        for _ in range(self.Ncells):
            self.atoms_ids.append([])
            self.neighbors1_cells_ids.append([])

        self.neighbors1_cells = np.zeros(self.Ncells, dtype=int)

        if self.long_range_flag == 1:
            self.next_neighbor = 1

        if self.next_neighbor == 1:
            self.neighbors2_cells = np.zeros(self.Ncells, dtype=int)
            for _ in range(self.Ncells):
                self.neighbors2_cells_ids.append([])

        self.corner[0] = -0.5 * self.lx
        self.corner[1] = -0.5 * self.ly
        self.corner[2] = -0.5 * self.lz

    def calculateAtomsInsideOfCell(self):
        pos = np.zeros(3)
        for i in range(self.moleculeTarget.natoms):
            pos[0] = self.moleculeTarget.x[i]
            pos[1] = self.moleculeTarget.y[i]
            pos[2] = self.moleculeTarget.z[i]

            index = self.calculateIndex(pos)

            # The original code uses moleculeTarget.id[i], assuming id[i] == i in Python implementation
            # since we don't store id array explicitly (it was just i)
            atom_id = i
            self.atoms_ids[index].append(atom_id)

        for i in range(self.Ncells):
            self.atoms_inside_cell[i] = len(self.atoms_ids[i])

    def calculateIndex(self, pos):
        xi = pos[0] - self.corner[0]
        i = int(math.floor(xi / (self.lj_cutoff + self.skin)))

        yi = pos[1] - self.corner[1]
        j = int(math.floor(yi / (self.lj_cutoff + self.skin)))

        zi = pos[2] - self.corner[2]
        k = int(math.floor(zi / (self.lj_cutoff + self.skin)))

        return k + self.Nz * j + self.Nz * self.Ny * i

    def sortingAtoms(self):
        # Reorder molecule atoms based on cell structure
        # This is important for cache locality and efficient access in simulation

        natoms = self.moleculeTarget.natoms

        # Temporary storage
        new_atomName = []
        new_x = np.zeros(natoms)
        new_y = np.zeros(natoms)
        new_z = np.zeros(natoms)
        new_q = np.zeros(natoms)
        new_m = np.zeros(natoms)
        new_eps = np.zeros(natoms)
        new_sig = np.zeros(natoms)

        new_eps_central = None
        new_sig_central = None
        if self.gas_buffer_flag == 3:
            new_eps_central = np.zeros(natoms)
            new_sig_central = np.zeros(natoms)

        iatom = 0
        for i in range(self.Ncells):
            if self.atoms_inside_cell[i] > 0:
                for j in range(self.atoms_inside_cell[i]):
                    atom_id = self.atoms_ids[i][j]

                    new_atomName.append(self.moleculeTarget.atomName[atom_id])
                    new_x[iatom] = self.moleculeTarget.x[atom_id]
                    new_y[iatom] = self.moleculeTarget.y[atom_id]
                    new_z[iatom] = self.moleculeTarget.z[atom_id]
                    new_q[iatom] = self.moleculeTarget.q[atom_id]
                    new_m[iatom] = self.moleculeTarget.m[atom_id]
                    new_eps[iatom] = self.moleculeTarget.eps[atom_id]
                    new_sig[iatom] = self.moleculeTarget.sig[atom_id]

                    if self.gas_buffer_flag == 3:
                        new_eps_central[iatom] = self.moleculeTarget.eps_central[atom_id]
                        new_sig_central[iatom] = self.moleculeTarget.sig_central[atom_id]

                    iatom += 1

        # Copy back
        self.moleculeTarget.atomName = new_atomName
        self.moleculeTarget.x = new_x
        self.moleculeTarget.y = new_y
        self.moleculeTarget.z = new_z
        self.moleculeTarget.q = new_q
        self.moleculeTarget.m = new_m
        self.moleculeTarget.eps = new_eps
        self.moleculeTarget.sig = new_sig

        if self.gas_buffer_flag == 3:
            self.moleculeTarget.eps_central = new_eps_central
            self.moleculeTarget.sig_central = new_sig_central

        # Update atoms_ids to reflect new indices
        iatom = 0
        for i in range(self.Ncells):
            if self.atoms_inside_cell[i] > 0:
                for j in range(self.atoms_inside_cell[i]):
                    self.atoms_ids[i][j] = iatom
                    iatom += 1

        # Set head_atom_cell
        for i in range(self.Ncells):
            if self.atoms_inside_cell[i] > 0:
                self.head_atom_cell[i] = self.atoms_ids[i][0]
            else:
                self.head_atom_cell[i] = -1

    def calculateCellsNeighbors(self):
        indexes = np.zeros(3, dtype=int)
        d = self.lj_cutoff + self.skin
        R1 = self.lj_cutoff / d
        R2 = 0.0
        if self.next_neighbor == 1:
            R2 = self.coul_cutoff / d

        indexes_neighbors1 = []
        indexes_neighbors2 = []

        n1 = int(math.ceil(R1))
        ncells = n1

        if self.next_neighbor == 1:
            n2 = int(math.ceil(R2))
            ncells = n2

        for i in range(-ncells, ncells + 1):
            for j in range(-ncells, ncells + 1):
                for k in range(-ncells, ncells + 1):
                    indexes[0] = i
                    indexes[1] = j
                    indexes[2] = k

                    dist_sq = float(i*i + j*j + k*k)

                    is_neighbor1 = False
                    is_neighbor2 = False

                    # Logic from C++ simplified but essentially checking distance
                    # C++ code is verbose but seems to check if the cell is within R1 or R2 distance
                    # considering box shape approximation.

                    # Let's try to replicate the logic somewhat simpler but correctly.
                    # The C++ logic is very explicit about planes and axes to handle the 'sphere' of neighbors.

                    # Assuming basic distance check for now as translation of exact logic is tedious without copy-paste errors.
                    # Actually, for 1-to-1 I should try to match.

                    # Re-implementing logic based on C++ code structure for correctness
                    if i == 0 and j == 0 and k == 0:
                         indexes_neighbors1.append(indexes.copy())
                    elif i != 0 and j == 0 and k == 0: # axis x
                        if abs(i) <= n1:
                             indexes_neighbors1.append(indexes.copy())
                        elif self.next_neighbor == 1:
                             indexes_neighbors2.append(indexes.copy())
                    elif i == 0 and j != 0 and k == 0: # axis y
                        if abs(j) <= n1:
                             indexes_neighbors1.append(indexes.copy())
                        elif self.next_neighbor == 1:
                             indexes_neighbors2.append(indexes.copy())
                    elif i == 0 and j == 0 and k != 0: # axis z
                        if abs(k) <= n1:
                             indexes_neighbors1.append(indexes.copy())
                        elif self.next_neighbor == 1:
                             indexes_neighbors2.append(indexes.copy())
                    elif i == 0 and j != 0 and k != 0: # plane yz
                        val = float(j*j + k*k)
                        val_min = float((abs(j)-1)**2 + (abs(k)-1)**2)
                        if self.next_neighbor == 1 and val < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 1 and val_min < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 0:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())

                    elif i != 0 and j == 0 and k != 0: # plane xz
                        val = float(i*i + k*k)
                        val_min = float((abs(i)-1)**2 + (abs(k)-1)**2)
                        # Same logic structure...
                        if self.next_neighbor == 1 and val < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 1 and val_min < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 0:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())

                    elif i != 0 and j != 0 and k == 0: # plane xy
                        val = float(i*i + j*j)
                        val_min = float((abs(i)-1)**2 + (abs(j)-1)**2)
                        if self.next_neighbor == 1 and val < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 1 and val_min < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 0:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())

                    else: # 3D
                        val = float(i*i + j*j + k*k)
                        val_min = float((abs(i)-1)**2 + (abs(j)-1)**2 + (abs(k)-1)**2)
                        if self.next_neighbor == 1 and val < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 1 and val_min < R2*R2:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             else:
                                 indexes_neighbors2.append(indexes.copy())
                        elif self.next_neighbor == 0:
                             if val < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())
                             elif val_min < R1*R1:
                                 indexes_neighbors1.append(indexes.copy())

        # Populate neighbor lists
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    idx_cell = k + self.Nz * j + self.Nz * self.Ny * i

                    # Neighbors 1
                    for offset in indexes_neighbors1:
                        nx = i + offset[0]
                        ny = j + offset[1]
                        nz = k + offset[2]

                        if 0 <= nx < self.Nx and 0 <= ny < self.Ny and 0 <= nz < self.Nz:
                            idx_neighbor = nz + self.Nz * ny + self.Nz * self.Ny * nx
                            if self.atoms_inside_cell[idx_neighbor] != 0:
                                self.neighbors1_cells_ids[idx_cell].append(idx_neighbor)

                    if self.next_neighbor == 1:
                        # Neighbors 2
                         for offset in indexes_neighbors2:
                            nx = i + offset[0]
                            ny = j + offset[1]
                            nz = k + offset[2]

                            if 0 <= nx < self.Nx and 0 <= ny < self.Ny and 0 <= nz < self.Nz:
                                idx_neighbor = nz + self.Nz * ny + self.Nz * self.Ny * nx
                                if self.atoms_inside_cell[idx_neighbor] != 0:
                                    self.neighbors2_cells_ids[idx_cell].append(idx_neighbor)

        # Set sizes
        for i in range(self.Ncells):
            self.neighbors1_cells[i] = len(self.neighbors1_cells_ids[i])
            if self.next_neighbor == 1:
                self.neighbors2_cells[i] = len(self.neighbors2_cells_ids[i])

    def print_info(self):
        filled_cells = 0
        empty_cells = 0
        average_atoms_cells = 0
        maximum_atoms_cells = 0
        minimum_atoms_cells = self.moleculeTarget.natoms

        for i in range(self.Ncells):
            count = self.atoms_inside_cell[i]
            if count == 0:
                empty_cells += 1
            else:
                average_atoms_cells += count
                filled_cells += 1
                if count > maximum_atoms_cells:
                    maximum_atoms_cells = count
                if count < minimum_atoms_cells:
                    minimum_atoms_cells = count

        avg = float(average_atoms_cells) / float(filled_cells) if filled_cells > 0 else 0

        print("*********************************************************")
        print("Linked-cell: ")
        print(f"Numbers of cells: {self.Ncells}")
        print(f"Nx: {self.Nx} Ny: {self.Ny} Nz: {self.Nz}")
        print(f"Filled cells: {filled_cells}")
        print(f"Empty cells: {empty_cells}")
        print(f"Average atoms per cell: {avg:g}")
        print(f"Maximum atoms per cell: {maximum_atoms_cells}")
        print(f"Minimum atoms per cell: {minimum_atoms_cells}")
        print("Simulation box: ")
        print(f"lx: {self.lx:g} Ang")
        print(f"ly: {self.ly:g} Ang")
        print(f"lz: {self.lz:g} Ang")
