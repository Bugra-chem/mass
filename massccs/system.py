import time
import math
import numpy as np
import taichi as ti
import sys
from massccs.input_params import Input
from massccs.molecule_target import MoleculeTarget
from massccs.gas_buffer import GasBuffer
from massccs.linked_cell import LinkedCell
from massccs.constants import *
from massccs.math_utils import *
from massccs.force import lennardjones_LC, lennardjones_coulomb_LC, lennardjones_induced_dipole_LC

# Initialize Taichi with double precision
ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.data_oriented
class System:
    def __init__(self, input_file):
        start = time.time()

        self.input = Input(input_file)
        self.input.printReadInput()

        self.nProbe = self.input.nProbe
        self.nIter = self.input.nIter
        self.seed = self.input.seed
        self.nthreads = self.input.nthreads
        self.targetFilename = self.input.targetFilename
        self.dt = self.input.dt
        self.temperatureTarget = self.input.temperatureTarget
        self.gas_buffer_flag = self.input.gas_buffer_flag
        self.equipotential_flag = self.input.equipotential_flag
        self.skin = self.input.skin
        self.short_range_cutoff = self.input.short_range_cutoff
        self.lj_cutoff = self.input.lj_cutoff
        self.long_range_flag = self.input.long_range_flag
        self.long_range_cutoff = self.input.long_range_cutoff
        self.coul_cutoff = self.input.coul_cutoff
        self.polarizability_flag = self.input.polarizability_flag
        self.alpha = self.input.alpha
        self.user_ff_flag = self.input.user_ff_flag
        self.user_ff = self.input.user_ff

        self.determine_force_type()

        self.gas = GasBuffer(self.gas_buffer_flag)

        start_molecule = time.time()
        self.moleculeTarget = MoleculeTarget(self.targetFilename, self.gas_buffer_flag, self.user_ff, self.user_ff_flag, self.force_type)
        end_molecule = time.time()
        print(f"orientation time of molecule target: {end_molecule - start_molecule:.7f} s")

        self.calculate_reduced_mass()

        # Convert alpha units to kcal/mol
        self.alpha *= ALPHA_TO_KCAL_MOL

        start_ellipsoid = time.time()
        if self.equipotential_flag == 1:
            print("Equipotential not implemented. Using geometric ellipsoid.")
            self.geometric_ellipsoid()
        else:
            self.geometric_ellipsoid()
        end_ellipsoid = time.time()
        print(f"ellipsoid calculation time: {end_ellipsoid - start_ellipsoid:.4e}s")

        self.bmax = max(self.a, self.b, self.c)
        print(f"maximal impact parameter: {self.bmax:.4f} Ang")

        start_linked_cell = time.time()
        self.linkedcell = LinkedCell(self.moleculeTarget, self.a, self.b, self.c,
                                     self.lj_cutoff, self.skin, self.long_range_flag,
                                     self.long_range_cutoff, self.coul_cutoff, self.gas_buffer_flag)
        end_linked_cell = time.time()
        print(f"linked-cell calculation time: {end_linked_cell - start_linked_cell:.8f} s")

        self.lx = 0.5 * self.linkedcell.lx
        self.ly = 0.5 * self.linkedcell.ly
        self.lz = 0.5 * self.linkedcell.lz

        # Initialize Taichi fields
        self.init_taichi_fields()

        start_ccs = time.time()

        # Run Simulation
        self.run_simulation_kernel()

        end_ccs = time.time()
        print(f"CCS time: {end_ccs - start_ccs:.4f} s")

        self.calculate_statistics()

        end = time.time()
        print(f"Total time: {end - start:.4f} s")
        print("Program finished...")

    def determine_force_type(self):
        if self.gas_buffer_flag in [1, 4, 5]: # He, Ar, co2
            if self.short_range_cutoff == 0 and self.long_range_flag == 0:
                self.force_type = 1
            elif self.short_range_cutoff == 1 and self.long_range_flag == 0:
                self.force_type = 2
            elif self.short_range_cutoff == 0 and self.polarizability_flag == 1 and self.long_range_cutoff == 0:
                self.force_type = 3
            elif self.short_range_cutoff == 1 and self.polarizability_flag == 1 and self.long_range_cutoff == 1:
                self.force_type = 4
            else:
                self.force_type = 2
        elif self.gas_buffer_flag in [2, 3]: # N2, CO2
            if self.short_range_cutoff == 0 and self.long_range_flag == 0 and self.polarizability_flag == 0:
                self.force_type = 1
            elif self.short_range_cutoff == 1 and self.long_range_flag == 0 and self.polarizability_flag == 0:
                self.force_type = 2
            elif self.short_range_cutoff == 0 and self.long_range_flag == 1 and self.long_range_cutoff == 0 and self.polarizability_flag == 0:
                self.force_type = 3
            elif self.short_range_cutoff == 1 and self.long_range_flag == 1 and self.long_range_cutoff == 1 and self.polarizability_flag == 0:
                self.force_type = 4
            elif self.short_range_cutoff == 0 and self.long_range_flag == 1 and self.long_range_cutoff == 0 and self.polarizability_flag == 1:
                self.force_type = 5
            elif self.short_range_cutoff == 1 and self.long_range_flag == 1 and self.long_range_cutoff == 1 and self.polarizability_flag == 1:
                self.force_type = 6
            else:
                self.force_type = 2

    def calculate_reduced_mass(self):
        if self.gas_buffer_flag in [1, 4, 5]:
            self.mu = self.moleculeTarget.mass * self.gas.mass / (self.moleculeTarget.mass + self.gas.mass)
            print(f"target mass: {self.moleculeTarget.mass} amu")
            print(f"gas mass: {self.gas.mass} amu")
            print(f"reduce mass: {self.mu} amu")
            print(f"charge state: {self.moleculeTarget.Q} e")
        elif self.gas_buffer_flag in [2, 3]:
            self.mu = self.gas.mass
            print(f"target mass: {self.moleculeTarget.mass} amu")
            print(f"gas mass: {self.gas.mass} amu")
            print(f"reduce mass: {self.mu} amu")
            print(f"charge state: {self.moleculeTarget.Q} e")
            self.Inertia = 0.0
            for i in range(self.gas.natoms):
                self.Inertia += self.gas.m[i] * (self.gas.z[i]**2)
            self.d_bond = self.gas.d

    def geometric_ellipsoid(self):
        maxX = abs(self.moleculeTarget.maxX)
        maxY = abs(self.moleculeTarget.maxY)
        maxZ = abs(self.moleculeTarget.maxZ)
        d = self.gas.d

        radius = 2.0 * self.coul_cutoff if self.long_range_flag == 1 else 2.0 * self.lj_cutoff

        print("*********************************************************")
        print("Geometric Ellipsoid: ")
        print("*********************************************************")
        print(f"maximal distances: {maxX:g}  {maxY:g}  {maxZ:g}  Ang")

        self.a = maxX + radius + self.skin + d
        self.b = maxY + radius + self.skin + d
        self.c = maxZ + radius + self.skin + d

        print(f"Initial axis length: {self.a:g}  {self.b:g}  {self.c:g}  Ang")

        delta = 0.01
        outside = True

        while outside:
            outside = False
            for i in range(self.moleculeTarget.natoms):
                xi = self.moleculeTarget.x[i]
                yi = self.moleculeTarget.y[i]
                zi = self.moleculeTarget.z[i]
                ri = math.sqrt(xi*xi + yi*yi + zi*zi)
                if ri > 0:
                    ux = xi/ri
                    uy = yi/ri
                    uz = zi/ri
                else:
                    ux, uy, uz = 0, 0, 0

                x = xi + (radius+d)*ux
                y = yi + (radius+d)*uy
                z = zi + (radius+d)*uz

                ellipsoid = (x/self.a)**2 + (y/self.b)**2 + (z/self.c)**2

                if ellipsoid > 1.0:
                    self.a += delta
                    self.b += delta
                    self.c += delta
                    outside = True
                    break

        print(f"Ellipsoid axis length: {self.a:g}  {self.b:g}  {self.c:g}  Ang")

    def init_taichi_fields(self):
        self.total_simulations = self.nIter * self.nProbe

        self.dOmega_vec = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.Nscatter_vec = ti.field(dtype=ti.i32, shape=self.total_simulations)
        self.Nfree_vec = ti.field(dtype=ti.i32, shape=self.total_simulations)
        self.Nlost_vec = ti.field(dtype=ti.i32, shape=self.total_simulations)

        self.rnd_vec1 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec2 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec3 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec4 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec5 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec6 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec7 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec8 = ti.field(dtype=ti.f64, shape=self.total_simulations)
        self.rnd_vec9 = ti.field(dtype=ti.f64, shape=self.total_simulations)

        np.random.seed(self.seed)
        self.rnd_vec1.from_numpy(np.sqrt(self.bmax * self.bmax * np.random.rand(self.total_simulations)))
        self.rnd_vec2.from_numpy(np.random.rand(self.total_simulations))
        self.rnd_vec3.from_numpy(np.random.rand(self.total_simulations))
        self.rnd_vec4.from_numpy(np.random.rand(self.total_simulations))
        self.rnd_vec5.from_numpy(np.random.rand(self.total_simulations))
        if self.gas_buffer_flag in [2, 3]:
            self.rnd_vec6.from_numpy(np.random.rand(self.total_simulations))
            self.rnd_vec7.from_numpy(np.random.rand(self.total_simulations))
            self.rnd_vec8.from_numpy(np.random.rand(self.total_simulations))
            self.rnd_vec9.from_numpy(np.random.rand(self.total_simulations))

        natoms = self.moleculeTarget.natoms
        self.ti_target_x = ti.field(dtype=ti.f64, shape=natoms)
        self.ti_target_y = ti.field(dtype=ti.f64, shape=natoms)
        self.ti_target_z = ti.field(dtype=ti.f64, shape=natoms)
        self.ti_target_q = ti.field(dtype=ti.f64, shape=natoms)
        self.ti_target_eps = ti.field(dtype=ti.f64, shape=natoms)
        self.ti_target_sig = ti.field(dtype=ti.f64, shape=natoms)

        if self.gas_buffer_flag == 3:
             self.ti_target_eps_central = ti.field(dtype=ti.f64, shape=natoms)
             self.ti_target_sig_central = ti.field(dtype=ti.f64, shape=natoms)
             self.ti_target_eps_central.from_numpy(self.moleculeTarget.eps_central)
             self.ti_target_sig_central.from_numpy(self.moleculeTarget.sig_central)

        self.ti_target_x.from_numpy(self.moleculeTarget.x)
        self.ti_target_y.from_numpy(self.moleculeTarget.y)
        self.ti_target_z.from_numpy(self.moleculeTarget.z)
        self.ti_target_q.from_numpy(self.moleculeTarget.q)
        self.ti_target_eps.from_numpy(self.moleculeTarget.eps)
        self.ti_target_sig.from_numpy(self.moleculeTarget.sig)

        ncells = self.linkedcell.Ncells

        max_atoms = 0
        for i in range(ncells):
            max_atoms = max(max_atoms, self.linkedcell.atoms_inside_cell[i])

        if max_atoms == 0:
            max_atoms = 1 # avoid 0 size field

        self.ti_atoms_inside_cell = ti.field(dtype=ti.i32, shape=ncells)
        self.ti_atoms_inside_cell.from_numpy(self.linkedcell.atoms_inside_cell)

        self.ti_cell_atoms = ti.field(dtype=ti.i32, shape=(ncells, max_atoms))

        cell_atoms_np = np.zeros((ncells, max_atoms), dtype=np.int32)
        for i in range(ncells):
            count = self.linkedcell.atoms_inside_cell[i]
            for j in range(count):
                cell_atoms_np[i, j] = self.linkedcell.atoms_ids[i][j]
        self.ti_cell_atoms.from_numpy(cell_atoms_np)

        max_neighbors1 = 0
        for i in range(ncells):
            max_neighbors1 = max(max_neighbors1, len(self.linkedcell.neighbors1_cells_ids[i]))
        if max_neighbors1 == 0: max_neighbors1 = 1

        self.ti_neighbors1_count = ti.field(dtype=ti.i32, shape=ncells)
        self.ti_neighbors1_count.from_numpy(self.linkedcell.neighbors1_cells)

        self.ti_neighbors1 = ti.field(dtype=ti.i32, shape=(ncells, max_neighbors1))
        neighbors1_np = np.zeros((ncells, max_neighbors1), dtype=np.int32)
        for i in range(ncells):
            count = len(self.linkedcell.neighbors1_cells_ids[i])
            for j in range(count):
                neighbors1_np[i, j] = self.linkedcell.neighbors1_cells_ids[i][j]
        self.ti_neighbors1.from_numpy(neighbors1_np)

        max_neighbors2 = 1
        if self.linkedcell.next_neighbor == 1:
            max_neighbors2 = 0
            for i in range(ncells):
                max_neighbors2 = max(max_neighbors2, len(self.linkedcell.neighbors2_cells_ids[i]))
            if max_neighbors2 == 0: max_neighbors2 = 1

            self.ti_neighbors2_count = ti.field(dtype=ti.i32, shape=ncells)
            self.ti_neighbors2_count.from_numpy(self.linkedcell.neighbors2_cells)

            self.ti_neighbors2 = ti.field(dtype=ti.i32, shape=(ncells, max_neighbors2))
            neighbors2_np = np.zeros((ncells, max_neighbors2), dtype=np.int32)
            for i in range(ncells):
                count = len(self.linkedcell.neighbors2_cells_ids[i])
                for j in range(count):
                    neighbors2_np[i, j] = self.linkedcell.neighbors2_cells_ids[i][j]
            self.ti_neighbors2.from_numpy(neighbors2_np)
        else:
            # Dummy fields for neighbor 2
            self.ti_neighbors2_count = ti.field(dtype=ti.i32, shape=ncells)
            self.ti_neighbors2 = ti.field(dtype=ti.i32, shape=(ncells, max_neighbors2))

    def run_simulation_kernel(self):
        print("*********************************************************")
        print("Trajectory calculations ")
        print("*********************************************************")

        if self.gas_buffer_flag in [1, 4, 5]:
            self.run_He_kernel(self.total_simulations)
        elif self.gas_buffer_flag == 2:
            raise NotImplementedError("Nitrogen (N2) simulation not yet implemented in Taichi port.")
        elif self.gas_buffer_flag == 3:
            raise NotImplementedError("Carbon Dioxide (CO2) simulation not yet implemented in Taichi port.")

    @ti.func
    def velGenerator(self, m: float, temperature: float, rnd: float) -> float:
        # Inverse CDF sampling for velocity distribution
        v = 0.0
        dv = 1.0E-5 # from C++
        sumProbability = 0.0

        # Constants for velDistr
        m_kg = m * AMU_TO_KG

        # Factor from C++: pow((m / (2.0 * BOLTZMANN_K * temperature)), 3.0)
        factor1 = (m_kg / (2.0 * BOLTZMANN_K * temperature))**3

        count = 0
        while sumProbability < rnd and count < 1000000: # Safety break
            # velDistr logic
            v_ms = v * ANG_TO_M / FS_TO_S

            term1 = v_ms**5
            term2 = factor1
            term3 = ti.exp(-(m_kg * v_ms**2) / (2.0 * BOLTZMANN_K * temperature))

            prob = term1 * term2 * term3
            sumProbability += prob
            v += dv
            count += 1

        return v

    @ti.func
    def KineticEnergy(self, m: float, v: ti.types.vector(3, float)) -> float:
        # 0.5 * m * v^2 * conversion
        return 0.5 * m * v.dot(v) * AMU_ANG_FS2_TO_KCAL_MOL

    @ti.kernel
    def run_He_kernel(self, n_simulations: int):
        # Grid parameters for force calculation
        d_cell = self.lj_cutoff + self.skin
        corner = ti.Vector([self.linkedcell.corner[0], self.linkedcell.corner[1], self.linkedcell.corner[2]])

        for j in range(n_simulations):
            # Setup
            hit = False
            success = False
            chi = 0.0

            # Initial random values
            rnd1 = self.rnd_vec1[j] # bi
            rnd2 = self.rnd_vec2[j] # velocity sample
            rnd3 = self.rnd_vec3[j] # gamma
            rnd4 = self.rnd_vec4[j] # phi
            rnd5 = self.rnd_vec5[j] # theta

            # Setup Probe
            bi = rnd1
            rcm = ti.Vector([bi, 0.0, self.bmax])

            vi_mag = self.velGenerator(self.mu, self.temperatureTarget, rnd2)
            vcm = ti.Vector([0.0, 0.0, -vi_mag])

            # Rotation
            gamma = 2.0 * math.pi * rnd3
            phi = ti.asin(2.0 * rnd4 - 1.0) + 0.5 * math.pi
            theta = 2.0 * math.pi * rnd5

            # Rotate rcm and vcm
            # Rz(gamma)
            x = rcm[0]; y = rcm[1]
            rcm[0] = x * ti.cos(gamma) - y * ti.sin(gamma)
            rcm[1] = x * ti.sin(gamma) + y * ti.cos(gamma)

            x = vcm[0]; y = vcm[1]
            vcm[0] = x * ti.cos(gamma) - y * ti.sin(gamma)
            vcm[1] = x * ti.sin(gamma) + y * ti.cos(gamma)

            # Ry(phi)
            x = rcm[0]; z = rcm[2]
            rcm[0] = x * ti.cos(phi) + z * ti.sin(phi)
            rcm[2] = x * -ti.sin(phi) + z * ti.cos(phi)

            x = vcm[0]; z = vcm[2]
            vcm[0] = x * ti.cos(phi) + z * ti.sin(phi)
            vcm[2] = x * -ti.sin(phi) + z * ti.cos(phi)

            # Rz(theta)
            x = rcm[0]; y = rcm[1]
            rcm[0] = x * ti.cos(theta) - y * ti.sin(theta)
            rcm[1] = x * ti.sin(theta) + y * ti.cos(theta)

            x = vcm[0]; y = vcm[1]
            vcm[0] = x * ti.cos(theta) - y * ti.sin(theta)
            vcm[1] = x * ti.sin(theta) + y * ti.cos(theta)

            # Ellipsoid intersection check
            ur = ti.Vector([rcm[0]/self.a, rcm[1]/self.b, rcm[2]/self.c])
            uv = ti.Vector([vcm[0]/self.a, vcm[1]/self.b, vcm[2]/self.c])

            cross = ur.cross(uv)
            delta = -cross.dot(cross) + uv.dot(uv)

            if delta >= 0:
                hit = True
                lambda_val = -ur.dot(uv)/uv.dot(uv) - ti.sqrt(delta)/uv.dot(uv)
                d_vec = lambda_val * vcm
                rcm += d_vec

            if hit:
                # Run Dynamics

                dt = self.dt
                # Store initial state for scattering angle
                vcm_i = vcm
                rcm_old = rcm

                # Force calculation variables
                force_vec = ti.Vector([0.0, 0.0, 0.0])
                Up = 0.0

                # Initial Force
                if self.force_type == 2:
                    force_vec, Up = lennardjones_LC(rcm, force_vec, self.lj_cutoff,
                                                    self.ti_atoms_inside_cell, self.ti_cell_atoms,
                                                    self.ti_neighbors1_count, self.ti_neighbors1,
                                                    self.ti_target_x, self.ti_target_y, self.ti_target_z,
                                                    self.ti_target_eps, self.ti_target_sig,
                                                    self.lx, self.ly, self.lz, corner, d_cell,
                                                    self.linkedcell.Nx, self.linkedcell.Ny, self.linkedcell.Nz)
                elif self.force_type == 4:
                    force_vec, Up = lennardjones_induced_dipole_LC(rcm, force_vec, self.lj_cutoff,
                                                    self.coul_cutoff, self.alpha,
                                                    self.ti_atoms_inside_cell, self.ti_cell_atoms,
                                                    self.ti_neighbors1_count, self.ti_neighbors1,
                                                    self.ti_neighbors2_count, self.ti_neighbors2,
                                                    self.ti_target_x, self.ti_target_y, self.ti_target_z,
                                                    self.ti_target_eps, self.ti_target_sig, self.ti_target_q,
                                                    self.lx, self.ly, self.lz, corner, d_cell,
                                                    self.linkedcell.Nx, self.linkedcell.Ny, self.linkedcell.Nz,
                                                    self.linkedcell.next_neighbor)

                fi = force_vec
                Ek = self.KineticEnergy(self.mu, vcm)
                Ei = Up + Ek

                trajTries = 0
                maxTries = 8
                stepcount = 0

                sim_running = True

                while sim_running and trajTries < maxTries and maxTries < 10:
                    # Verlet Integration

                    # First half
                    acc = 0.5 * dt / self.mu * KCALMOLANGAMU_TO_ANGFS2
                    vcm += acc * force_vec
                    rcm += dt * vcm

                    # Force calculation at new position
                    if self.force_type == 2:
                        force_vec, Up = lennardjones_LC(rcm, force_vec, self.lj_cutoff,
                                                    self.ti_atoms_inside_cell, self.ti_cell_atoms,
                                                    self.ti_neighbors1_count, self.ti_neighbors1,
                                                    self.ti_target_x, self.ti_target_y, self.ti_target_z,
                                                    self.ti_target_eps, self.ti_target_sig,
                                                    self.lx, self.ly, self.lz, corner, d_cell,
                                                    self.linkedcell.Nx, self.linkedcell.Ny, self.linkedcell.Nz)
                    elif self.force_type == 4:
                        force_vec, Up = lennardjones_induced_dipole_LC(rcm, force_vec, self.lj_cutoff,
                                                    self.coul_cutoff, self.alpha,
                                                    self.ti_atoms_inside_cell, self.ti_cell_atoms,
                                                    self.ti_neighbors1_count, self.ti_neighbors1,
                                                    self.ti_neighbors2_count, self.ti_neighbors2,
                                                    self.ti_target_x, self.ti_target_y, self.ti_target_z,
                                                    self.ti_target_eps, self.ti_target_sig, self.ti_target_q,
                                                    self.lx, self.ly, self.lz, corner, d_cell,
                                                    self.linkedcell.Nx, self.linkedcell.Ny, self.linkedcell.Nz,
                                                    self.linkedcell.next_neighbor)

                    # Second half
                    vcm += acc * force_vec

                    stepcount += 1

                    # Check boundary conditions
                    # box
                    if abs(rcm[0]) >= self.lx or abs(rcm[1]) >= self.ly or abs(rcm[2]) >= self.lz:
                        # Outside box
                        Ek = self.KineticEnergy(self.mu, vcm)
                        E = Up + Ek
                        dE = abs(E - Ei) / Ei * 100.0

                        if dE > 0.5:
                            # Energy conservation failed
                            trajTries += 1
                            maxTries += 1
                            dt = self.dt / (trajTries + 1)
                            stepcount = 0
                            sim_running = False # Fail
                            success = False
                        else:
                            # Success
                            vcm_f = vcm
                            chi = angle_vec(vcm_i, vcm_f)
                            success = True
                            sim_running = False
                    else:
                        # Check ellipsoid
                        ur_x = rcm[0]/self.a
                        ur_y = rcm[1]/self.b
                        ur_z = rcm[2]/self.c
                        r_sq = ur_x**2 + ur_y**2 + ur_z**2

                        if r_sq > 1.0:
                             # Outside ellipsoid
                            Ek = self.KineticEnergy(self.mu, vcm)
                            E = Up + Ek
                            dE = abs(E - Ei) / Ei * 100.0

                            if dE > 0.5:
                                trajTries += 1
                                maxTries += 1
                                dt = self.dt / (trajTries + 1)
                                stepcount = 0
                                sim_running = False
                                success = False
                            else:
                                vcm_f = vcm
                                chi = angle_vec(vcm_i, vcm_f)
                                success = True
                                sim_running = False

                if success:
                    self.dOmega_vec[j] = math.pi * (1.0 - ti.cos(chi)) * (self.bmax * self.bmax)
                    self.Nscatter_vec[j] = 1
                else:
                    self.Nlost_vec[j] = 1
            else:
                self.Nfree_vec[j] = 1

    @ti.kernel
    def run_N2_kernel(self, n_simulations: int):
        # NotImplementedError is not supported in Taichi kernel
        pass

    @ti.kernel
    def run_CO2_kernel(self, n_simulations: int):
        # NotImplementedError is not supported in Taichi kernel
        pass

    def calculate_statistics(self):
        # Collect results from fields
        dOmega_vec = self.dOmega_vec.to_numpy()
        Nscatter_vec = self.Nscatter_vec.to_numpy()
        Nfree_vec = self.Nfree_vec.to_numpy()
        Nlost_vec = self.Nlost_vec.to_numpy()

        count = 0
        Omega = 0.0
        Omega2 = 0.0

        for i in range(self.nIter):
            dOmega = 0.0
            Nscatter = 0
            Nfree = 0
            Nlost = 0

            for j in range(self.nProbe):
                dOmega += dOmega_vec[count]
                Nscatter += Nscatter_vec[count]
                Nfree += Nfree_vec[count]
                Nlost += Nlost_vec[count]
                count += 1

            omega_val = (1.0 / (Nscatter + Nfree)) * dOmega if (Nscatter + Nfree) > 0 else 0
            Omega += omega_val
            Omega2 += omega_val**2

            print(f"Ntraj: {self.nProbe}")
            print(f"Nfree: {Nfree}")
            print(f"Nscatter: {Nscatter}")
            print(f"Nlost: {Nlost}")
            print(f"omega: {omega_val:g}")

        self.CCS_ave = Omega / self.nIter
        sig2 = Omega2 / self.nIter - self.CCS_ave**2
        self.CCS_err = math.sqrt(abs(sig2) / self.nIter)

        print("*********************************************************")
        print(f"average value of CCS = {self.CCS_ave:g} Ang^2")
        print(f"error value of CCS = {self.CCS_err:g} Ang^2")
