import numpy as np
import math
import taichi as ti

@ti.data_oriented
class LinkedCell:
    def __init__(self, molecule_target, a, b, c, lj_cutoff, skin, long_range_flag, long_range_cutoff, coul_cutoff, gas_buffer_flag):
        self.molecule_target = molecule_target
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

        self.corner = np.zeros(3)

        self.calculate_number_of_cells()
        self.calculate_atoms_inside_of_cell()
        self.sorting_atoms()
        self.calculate_cells_neighbors()
        self.print_info()

        # Ti fields for neighbors (will be populated in System)
        # We need to expose data for Taichi kernels
        # Using dense arrays for simplicity, or sparse if needed.
        # For now, keeping data in numpy/python objects and maybe moving to fields later if strictly needed for performance
        # but the main simulation loop will be in Taichi, so we will need to transfer this data.

    def calculate_number_of_cells(self):
        # number of cell for each axis
        self.Nx = int(math.ceil(self.lx / (self.lj_cutoff + self.skin)))
        self.Ny = int(math.ceil(self.ly / (self.lj_cutoff + self.skin)))
        self.Nz = int(math.ceil(self.lz / (self.lj_cutoff + self.skin)))

        # box simulation domain
        self.lx = float(self.Nx) * (self.lj_cutoff + self.skin)
        self.ly = float(self.Ny) * (self.lj_cutoff + self.skin)
        self.lz = float(self.Nz) * (self.lj_cutoff + self.skin)

        self.Ncells = self.Nx * self.Ny * self.Nz

        self.atoms_inside_cell = np.zeros(self.Ncells, dtype=int)
        self.head_atom_cell = np.zeros(self.Ncells, dtype=int)
        self.atoms_ids = [[] for _ in range(self.Ncells)]
        self.neighbors1_cells = np.zeros(self.Ncells, dtype=int)
        self.neighbors1_cells_ids = [[] for _ in range(self.Ncells)]

        if self.long_range_flag == 1:
            self.next_neighbor = 1

        if self.next_neighbor == 1:
            self.neighbors2_cells = np.zeros(self.Ncells, dtype=int)
            self.neighbors2_cells_ids = [[] for _ in range(self.Ncells)]
        else:
            self.neighbors2_cells = None
            self.neighbors2_cells_ids = None

        self.corner[0] = -0.5 * self.lx
        self.corner[1] = -0.5 * self.ly
        self.corner[2] = -0.5 * self.lz

    def calculate_atoms_inside_of_cell(self):
        for i in range(self.molecule_target.natoms):
            pos = np.array([self.molecule_target.x[i], self.molecule_target.y[i], self.molecule_target.z[i]])
            index = self.calculate_index(pos)

            id_val = self.molecule_target.id[i]
            self.atoms_ids[index].append(id_val)

        for i in range(self.Ncells):
            self.atoms_inside_cell[i] = len(self.atoms_ids[i])

    def calculate_index(self, pos):
        xi = pos[0] - self.corner[0]
        i = int(math.floor(xi / (self.lj_cutoff + self.skin)))
        yi = pos[1] - self.corner[1]
        j = int(math.floor(yi / (self.lj_cutoff + self.skin)))
        zi = pos[2] - self.corner[2]
        k = int(math.floor(zi / (self.lj_cutoff + self.skin)))

        # Clamp indices to be safe
        i = max(0, min(i, self.Nx - 1))
        j = max(0, min(j, self.Ny - 1))
        k = max(0, min(k, self.Nz - 1))

        index = k + self.Nz * j + self.Nz * self.Ny * i
        return index

    def sorting_atoms(self):
        natoms = self.molecule_target.natoms

        tmp_atomName = [""] * natoms
        tmp_x = np.zeros(natoms)
        tmp_y = np.zeros(natoms)
        tmp_z = np.zeros(natoms)
        tmp_q = np.zeros(natoms)
        tmp_m = np.zeros(natoms)
        tmp_eps = np.zeros(natoms)
        tmp_sig = np.zeros(natoms)
        tmp_eps_central = np.zeros(natoms)
        tmp_sig_central = np.zeros(natoms)

        iatom = 0
        for i in range(self.Ncells):
            if self.atoms_inside_cell[i] > 0:
                for j in range(self.atoms_inside_cell[i]):
                    id_val = self.atoms_ids[i][j]
                    tmp_atomName[iatom] = self.molecule_target.atomName[id_val]
                    tmp_x[iatom] = self.molecule_target.x[id_val]
                    tmp_y[iatom] = self.molecule_target.y[id_val]
                    tmp_z[iatom] = self.molecule_target.z[id_val]
                    tmp_q[iatom] = self.molecule_target.q[id_val]
                    tmp_m[iatom] = self.molecule_target.m[id_val]
                    tmp_eps[iatom] = self.molecule_target.eps[id_val]
                    tmp_sig[iatom] = self.molecule_target.sig[id_val]
                    if self.gas_buffer_flag == 3:
                        tmp_eps_central[iatom] = self.molecule_target.eps_central[id_val]
                        tmp_sig_central[iatom] = self.molecule_target.sig_central[id_val]
                    iatom += 1

        # Update molecule target
        for i in range(natoms):
            self.molecule_target.atomName[i] = tmp_atomName[i]
            self.molecule_target.x[i] = tmp_x[i]
            self.molecule_target.y[i] = tmp_y[i]
            self.molecule_target.z[i] = tmp_z[i]
            self.molecule_target.q[i] = tmp_q[i]
            self.molecule_target.m[i] = tmp_m[i]
            self.molecule_target.eps[i] = tmp_eps[i]
            self.molecule_target.sig[i] = tmp_sig[i]
            if self.gas_buffer_flag == 3:
                self.molecule_target.eps_central[i] = tmp_eps_central[i]
                self.molecule_target.sig_central[i] = tmp_sig_central[i]

        # Update IDs in cells to match new ordering (which is contiguous per cell)
        iatom = 0
        for i in range(self.Ncells):
            if self.atoms_inside_cell[i] > 0:
                for j in range(self.atoms_inside_cell[i]):
                    self.atoms_ids[i][j] = iatom
                    iatom += 1

        for i in range(self.Ncells):
            if self.atoms_inside_cell[i] > 0:
                self.head_atom_cell[i] = self.atoms_ids[i][0]
            else:
                self.head_atom_cell[i] = -1

    def calculate_cells_neighbors(self):
        d = self.lj_cutoff + self.skin
        R1 = self.lj_cutoff / d
        R2 = 0.0
        if self.next_neighbor == 1:
            R2 = self.coul_cutoff / d

        indexes_neighbors1 = []
        indexes_neighbors2 = []

        n1 = int(math.ceil(R1))
        ncells = n1
        n2 = 0
        if self.next_neighbor == 1:
            n2 = int(math.ceil(R2))
            ncells = n2

        # Logic to fill neighbors offsets
        # Simplifying translation of C++ logic by iterating and checking conditions
        for i in range(-ncells, ncells + 1):
            for j in range(-ncells, ncells + 1):
                for k in range(-ncells, ncells + 1):
                    indexes = [i, j, k]
                    dist_sq = float(i*i + j*j + k*k)

                    # Approximating the complex condition tree from C++
                    # which basically checks if cell is within radius, considering box geometry approximation?
                    # The C++ code distinguishes between full distance and distance to box faces/edges?
                    # Let's try to mimic the logic structure for correctness.

                    is_n1 = False
                    is_n2 = False

                    if i==0 and j==0 and k==0:
                         indexes_neighbors1.append(indexes)
                         continue

                    # The C++ code has many branches.
                    # It seems to check if the cell is within R1 (neighbor1) or R2 (neighbor2)
                    # using a mix of exact distance and projected distance depending on if on axis/plane.

                    # Let's reuse the logic structure slightly simplified or direct translation.

                    dist_sq_check = dist_sq # Default check

                    if self.next_neighbor == 1:
                        # Logic with 2 neighbors sets
                        # Check specific cases like C++

                        # Just implementing the critical part: distinguishing between neighbor sets
                        # If inside R1 -> n1, else if inside R2 -> n2

                        # However, C++ code computes distance differently for planes/axes.
                        # E.g. for (i!=0 && j==0 && k==0) (axis x), check abs(i) <= n1

                        added = False

                        # Axis checks
                        if j==0 and k==0: # X axis
                            if abs(i) <= n1: indexes_neighbors1.append(indexes)
                            else: indexes_neighbors2.append(indexes)
                            continue
                        if i==0 and k==0: # Y axis
                            if abs(j) <= n1: indexes_neighbors1.append(indexes)
                            else: indexes_neighbors2.append(indexes)
                            continue
                        if i==0 and j==0: # Z axis
                            if abs(k) <= n1: indexes_neighbors1.append(indexes)
                            else: indexes_neighbors2.append(indexes)
                            continue

                        # Plane checks
                        # Helper to check logic
                        def check_dist(d2, r1_sq, r2_sq, d2_alt):
                            if d2 < r2_sq:
                                if d2 < r1_sq: return 1
                                elif d2_alt < r1_sq: return 1
                                else: return 2
                            elif d2_alt < r2_sq:
                                if d2 < r1_sq: return 1 # Impossible if d2 >= r2_sq > r1_sq? C++ logic is weird here.
                                # Actually: if d2 >= R2^2, check d2_alt.
                                # C++: } else if (d2_alt < R2*R2) {
                                #   if (d2 < R1*R1) -> 1
                                #   else if (d2_alt < R1*R1) -> 1
                                #   else -> 2
                                if d2 < r1_sq: return 1
                                elif d2_alt < r1_sq: return 1
                                else: return 2
                            return 0

                        R1_sq = R1*R1
                        R2_sq = R2*R2

                        val = 0
                        if i==0: # YZ plane
                            d2 = float(j*j + k*k)
                            d2_alt = float((abs(j)-1)**2 + (abs(k)-1)**2)
                            val = check_dist(d2, R1_sq, R2_sq, d2_alt)
                        elif j==0: # XZ plane
                            d2 = float(i*i + k*k)
                            d2_alt = float((abs(i)-1)**2 + (abs(k)-1)**2)
                            val = check_dist(d2, R1_sq, R2_sq, d2_alt)
                        elif k==0: # XY plane
                            d2 = float(i*i + j*j)
                            d2_alt = float((abs(i)-1)**2 + (abs(j)-1)**2)
                            val = check_dist(d2, R1_sq, R2_sq, d2_alt)
                        else: # Bulk
                            d2 = float(i*i + j*j + k*k)
                            d2_alt = float((abs(i)-1)**2 + (abs(j)-1)**2 + (abs(k)-1)**2)
                            val = check_dist(d2, R1_sq, R2_sq, d2_alt)

                        if val == 1: indexes_neighbors1.append(indexes)
                        elif val == 2: indexes_neighbors2.append(indexes)

                    else:
                        # Only R1 check
                        # Simplified logic from C++ else block
                         # Axis checks
                        if j==0 and k==0: # X axis
                            if abs(i) <= n1: indexes_neighbors1.append(indexes)
                            continue
                        if i==0 and k==0: # Y axis
                            if abs(j) <= n1: indexes_neighbors1.append(indexes)
                            continue
                        if i==0 and j==0: # Z axis
                            if abs(k) <= n1: indexes_neighbors1.append(indexes)
                            continue

                        R1_sq = R1*R1
                        val = 0
                        if i==0: # YZ plane
                            d2 = float(j*j + k*k)
                            d2_alt = float((abs(j)-1)**2 + (abs(k)-1)**2)
                            if d2 < R1_sq or d2_alt < R1_sq: val = 1
                        elif j==0: # XZ plane
                            d2 = float(i*i + k*k)
                            d2_alt = float((abs(i)-1)**2 + (abs(k)-1)**2)
                            if d2 < R1_sq or d2_alt < R1_sq: val = 1
                        elif k==0: # XY plane
                            d2 = float(i*i + j*j)
                            d2_alt = float((abs(i)-1)**2 + (abs(j)-1)**2)
                            if d2 < R1_sq or d2_alt < R1_sq: val = 1
                        else: # Bulk
                            d2 = float(i*i + j*j + k*k)
                            d2_alt = float((abs(i)-1)**2 + (abs(j)-1)**2 + (abs(k)-1)**2)
                            if d2 < R1_sq or d2_alt < R1_sq: val = 1

                        if val == 1: indexes_neighbors1.append(indexes)

        # Populate neighbor lists for cells
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    idx_cell = k + self.Nz * j + self.Nz * self.Ny * i

                    # Neighbors 1
                    for neighbor in indexes_neighbors1:
                        ni = i + neighbor[0]
                        nj = j + neighbor[1]
                        nk = k + neighbor[2]

                        if 0 <= ni < self.Nx and 0 <= nj < self.Ny and 0 <= nk < self.Nz:
                            idx_neighbor = nk + self.Nz * nj + self.Nz * self.Ny * ni
                            if self.atoms_inside_cell[idx_neighbor] != 0:
                                self.neighbors1_cells_ids[idx_cell].append(idx_neighbor)

                    if self.next_neighbor == 1:
                        # Neighbors 2
                        for neighbor in indexes_neighbors2:
                            ni = i + neighbor[0]
                            nj = j + neighbor[1]
                            nk = k + neighbor[2]

                            if 0 <= ni < self.Nx and 0 <= nj < self.Ny and 0 <= nk < self.Nz:
                                idx_neighbor = nk + self.Nz * nj + self.Nz * self.Ny * ni
                                if self.atoms_inside_cell[idx_neighbor] != 0:
                                    self.neighbors2_cells_ids[idx_cell].append(idx_neighbor)

        for i in range(self.Ncells):
            self.neighbors1_cells[i] = len(self.neighbors1_cells_ids[i])
            if self.next_neighbor == 1:
                self.neighbors2_cells[i] = len(self.neighbors2_cells_ids[i])

    def print_info(self):
        filled_cells = 0
        empty_cells = 0
        average_atoms_cells = 0
        maximum_atoms_cells = 0
        minimum_atoms_cells = self.molecule_target.natoms

        for i in range(self.Ncells):
            if self.atoms_inside_cell[i] == 0:
                empty_cells += 1
            else:
                average_atoms_cells += self.atoms_inside_cell[i]
                filled_cells += 1
                if self.atoms_inside_cell[i] > maximum_atoms_cells:
                    maximum_atoms_cells = self.atoms_inside_cell[i]
                if self.atoms_inside_cell[i] < minimum_atoms_cells:
                    minimum_atoms_cells = self.atoms_inside_cell[i]

        avg = float(average_atoms_cells) / filled_cells if filled_cells > 0 else 0.0

        print("*********************************************************")
        print("Linked-cell: ")
        print(f"Numbers of cells: {self.Ncells}")
        print(f"Nx: {self.Nx} Ny: {self.Ny} Nz: {self.Nz}")
        print(f"Filled cells: {filled_cells}")
        print(f"Empty cells: {empty_cells}")
        print(f"Average atoms per cell: {avg}")
        print(f"Maximum atoms per cell: {maximum_atoms_cells}")
        print(f"Minimum atoms per cell: {minimum_atoms_cells}")
        print("Simulation box: ")
        print(f"lx: {self.lx} Ang")
        print(f"ly: {self.ly} Ang")
        print(f"lz: {self.lz} Ang")
