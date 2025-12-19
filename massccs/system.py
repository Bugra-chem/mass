import numpy as np
import taichi as ti
import math
import time
from .constants import *
from .input_parser import InputParser
from .molecule_target import MoleculeTarget
from .gas_buffer import GasBuffer
from .linked_cell import LinkedCell
from .force import Force
from .random_number import RandomNumber

@ti.data_oriented
class System:
    def __init__(self, input_filename):
        start_time = time.time()
        self.input = InputParser(input_filename)
        self.input.print_read_input()

        self.n_probe = self.input.n_probe
        self.n_iter = self.input.n_iter
        self.seed = self.input.seed
        self.mt = RandomNumber(self.seed)
        self.n_threads = self.input.n_threads
        self.target_filename = self.input.target_filename
        self.dt = self.input.dt
        self.temperature_target = self.input.temperature_target
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

        self.force_type = 2 # Default fallback
        if self.gas_buffer_flag in [1, 4, 5]: # He, Ar, co2 (point)
             if self.short_range_cutoff == 0 and self.long_range_flag == 0:
                 self.force_type = 1
             elif self.short_range_cutoff == 1 and self.long_range_flag == 0:
                 self.force_type = 2
             elif self.short_range_cutoff == 0 and self.polarizability_flag == 1 and self.long_range_cutoff == 0:
                 self.force_type = 3
             elif self.short_range_cutoff == 1 and self.polarizability_flag == 1 and self.long_range_cutoff == 1:
                 self.force_type = 4
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

        self.gas = GasBuffer(self.gas_buffer_flag)

        start_mol = time.time()
        self.molecule_target = MoleculeTarget(self.target_filename, self.gas_buffer_flag, self.user_ff, self.user_ff_flag, self.force_type)
        print(f"orientation time of molecule target: {time.time() - start_mol} s")

        if self.gas_buffer_flag in [1, 4, 5]:
            self.mu = self.molecule_target.mass * self.gas.mass / (self.molecule_target.mass + self.gas.mass)
            print(f"target mass: {self.molecule_target.mass} amu")
            print(f"gas mass: {self.gas.mass} amu")
            print(f"reduce mass: {self.mu} amu")
            print(f"charge state: {self.molecule_target.Q} e")
        elif self.gas_buffer_flag in [2, 3]:
            self.mu = self.gas.mass
            print(f"target mass: {self.molecule_target.mass} amu")
            print(f"gas mass: {self.gas.mass} amu")
            print(f"reduce mass: {self.mu} amu")
            print(f"charge state: {self.molecule_target.Q} e")
            self.Inertia = 0.0
            for i in range(self.gas.natoms):
                self.Inertia += self.gas.m[i] * (self.gas.z[i]**2)
            self.d_bond = self.gas.d

        self.alpha *= ALPHA_TO_KCAL_MOL

        start_ell = time.time()
        if self.equipotential_flag == 1:
            print("Equipotential not implemented yet.")
            # self.equipotential = ...
            pass
        else:
            self.geometric_ellipsoid()
        print(f"ellipsoid calculation time: {time.time() - start_ell} s")

        self.bmax = max(self.a, max(self.b, self.c))
        print(f"maximal impact parameter: {self.bmax} Ang")

        start_lc = time.time()
        self.linked_cell = LinkedCell(self.molecule_target, self.a, self.b, self.c,
                                      self.lj_cutoff, self.skin, self.long_range_flag,
                                      self.long_range_cutoff, self.coul_cutoff, self.gas_buffer_flag)
        print(f"linked-cell calculation time: {time.time() - start_lc} s")

        self.lx = 0.5 * self.linked_cell.lx
        self.ly = 0.5 * self.linked_cell.ly
        self.lz = 0.5 * self.linked_cell.lz

        # Initialize Taichi
        ti.init(arch=ti.cpu, default_fp=ti.f64) # Use CPU for consistency and reproducibility, matching C++ OpenMP logic roughly

        # Prepare Force object (Ti fields)
        self.force = Force(self.molecule_target, self.linked_cell, self.lj_cutoff, self.alpha, self.coul_cutoff)

        self.ccs_ave = 0.0
        self.ccs_err = 0.0

    def geometric_ellipsoid(self):
        maxX = abs(self.molecule_target.maxX)
        maxY = abs(self.molecule_target.maxY)
        maxZ = abs(self.molecule_target.maxZ)
        d = self.gas.d

        if self.long_range_flag == 1:
            radius = 2.0 * self.coul_cutoff
        else:
            radius = 2.0 * self.lj_cutoff

        print("*********************************************************")
        print("Geometric Ellipsoid: ")
        print("*********************************************************")
        print(f"maximal distances: {maxX}  {maxY}  {maxZ}  Ang")

        self.a = maxX + radius + self.skin + d
        self.b = maxY + radius + self.skin + d
        self.c = maxZ + radius + self.skin + d

        print(f"Initial axis length: {self.a}  {self.b}  {self.c}  Ang")

        delta = 0.01
        outside = True

        # Simplified ellipsoid fitting logic from C++
        # It expands ellipsoid until all atoms+radius are inside

        while outside:
            outside = False
            for i in range(self.molecule_target.natoms):
                xi = self.molecule_target.x[i]
                yi = self.molecule_target.y[i]
                zi = self.molecule_target.z[i]
                ri = math.sqrt(xi*xi + yi*yi + zi*zi)
                if ri > 1e-9:
                    ux = xi / ri
                    uy = yi / ri
                    uz = zi / ri
                else:
                    ux, uy, uz = 0, 0, 0

                x = xi + (radius + d) * ux
                y = yi + (radius + d) * uy
                z = zi + (radius + d) * uz

                ellipsoid = (x/self.a)**2 + (y/self.b)**2 + (z/self.c)**2

                if ellipsoid > 1.0:
                    self.a += delta
                    self.b += delta
                    self.c += delta
                    outside = True
                    break # Restart checking

        print(f"Ellipsoid axis length: {self.a}  {self.b}  {self.c}  Ang")

    def run_simulation(self):
        start_ccs = time.time()

        Niter = self.n_iter
        Ntraj = self.n_probe

        Omega = 0.0
        Omega2 = 0.0

        # Random numbers generation
        # In C++, it pre-generates random numbers for all trajectories
        # We can do the same or generate on fly in Taichi kernels if using Ti RNG.
        # But to match C++ 1-to-1 logic structure and potentially seed behavior (though diverse RNG implementations differ),
        # we'll generate them in Python and pass to Taichi.

        total_trajs = Niter * Ntraj
        rnd_vec1 = np.zeros(total_trajs)
        rnd_vec2 = np.zeros(total_trajs)
        rnd_vec3 = np.zeros(total_trajs)
        rnd_vec4 = np.zeros(total_trajs)
        rnd_vec5 = np.zeros(total_trajs)

        # Only generating for He for now as per instructions/focus
        for j in range(total_trajs):
            rnd_vec1[j] = math.sqrt((self.bmax**2) * self.mt.get_random_number())
            rnd_vec2[j] = self.mt.get_random_number()
            rnd_vec3[j] = self.mt.get_random_number()
            rnd_vec4[j] = self.mt.get_random_number()
            rnd_vec5[j] = self.mt.get_random_number()

        # Taichi Kernel for simulation
        # We need to map the logic of run_He (and others) to a kernel.
        # Since run_He has a while loop and updates state, it fits well in a Ti kernel running over trajectories.

        # Fields for results
        dOmega_vec = ti.field(dtype=ti.f64, shape=total_trajs)
        Nscatter_vec = ti.field(dtype=ti.i32, shape=total_trajs)
        Nfree_vec = ti.field(dtype=ti.i32, shape=total_trajs)
        Nlost_vec = ti.field(dtype=ti.i32, shape=total_trajs)

        # Fields for input randoms
        ti_rnd1 = ti.field(dtype=ti.f64, shape=total_trajs)
        ti_rnd2 = ti.field(dtype=ti.f64, shape=total_trajs)
        ti_rnd3 = ti.field(dtype=ti.f64, shape=total_trajs)
        ti_rnd4 = ti.field(dtype=ti.f64, shape=total_trajs)
        ti_rnd5 = ti.field(dtype=ti.f64, shape=total_trajs)

        ti_rnd1.from_numpy(rnd_vec1)
        ti_rnd2.from_numpy(rnd_vec2)
        ti_rnd3.from_numpy(rnd_vec3)
        ti_rnd4.from_numpy(rnd_vec4)
        ti_rnd5.from_numpy(rnd_vec5)

        print("*********************************************************")
        print("Trajectory calculations ")
        print("*********************************************************")

        if self.gas_buffer_flag in [1, 4, 5]:
             self.run_He_kernel(total_trajs, ti_rnd1, ti_rnd2, ti_rnd3, ti_rnd4, ti_rnd5,
                                dOmega_vec, Nscatter_vec, Nfree_vec, Nlost_vec)
        else:
             print("Only He/Ar/point-CO2 supported currently in this Python port step.")

        # Aggregate results
        # dOmega_vec etc are now populated.

        res_dOmega = dOmega_vec.to_numpy()
        res_Nscatter = Nscatter_vec.to_numpy()
        res_Nfree = Nfree_vec.to_numpy()
        res_Nlost = Nlost_vec.to_numpy()

        count = 0
        for i in range(Niter):
            dOmega = 0.0
            Nscatter = 0
            Nfree = 0
            Nlost = 0

            for j in range(Ntraj):
                dOmega += res_dOmega[count]
                Nscatter += res_Nscatter[count]
                Nfree += res_Nfree[count]
                Nlost += res_Nlost[count]
                count += 1

            omega_val = (1.0 / (Nscatter + Nfree)) * dOmega if (Nscatter + Nfree) > 0 else 0.0
            Omega += omega_val
            Omega2 += omega_val**2

            print(f"Ntraj: {Ntraj}")
            print(f"Nfree: {Nfree}")
            print(f"Nscatter: {Nscatter}")
            print(f"Nlost: {Nlost}")
            print(f"omega: {omega_val:g}")

        print(f"CCS time: {time.time() - start_ccs} s")

        self.ccs_ave = Omega / Niter
        sig2 = Omega2 / Niter - self.ccs_ave**2
        self.ccs_err = math.sqrt(abs(sig2) / Niter) # abs to avoid tiny neg from float precision

        print("*********************************************************")
        print(f"average value of CCS = {self.ccs_ave} Ang^2")
        print(f"error value of CCS = {self.ccs_err} Ang^2")
        print(f"Total time: {time.time() - start_ccs} s") # Using start_ccs as base for now

    @ti.kernel
    def run_He_kernel(self, n_trajs: ti.i32,
                      rnd1: ti.template(), rnd2: ti.template(), rnd3: ti.template(), rnd4: ti.template(), rnd5: ti.template(),
                      dOmega: ti.template(), Nscatter: ti.template(), Nfree: ti.template(), Nlost: ti.template()):

        for i in range(n_trajs):
            # Setup
            bi = rnd1[i]

            # Initial position
            rcm = ti.Vector([bi, 0.0, self.bmax])

            # Velocity generator
            # velDistr logic inversion is complex to do inside kernel without pre-table or iterative solve.
            # But the C++ code does iterative solve.
            # "velGenerator(double m, double temperature, double sd)"
            # Let's inline a simplified version or assume standard distribution logic if possible,
            # but strict copy means implementing the rejection/inverse transform sampling.
            # C++ uses rejection? No, it integrates PDF until sum > sd (random [0,1]).
            # This is inverse transform sampling on discretized PDF?
            # "v += dv; sumProbability += velDistr..."
            # Yes.

            m_kg = self.gas.m[0] * AMU_TO_KG
            # velDistr params
            # v in Ang/fs -> m/s
            # T

            v = 0.0
            dv = 1.0E-5
            sumProbability = 0.0
            target_prob = rnd2[i]

            # This loop might be slow in kernel, but let's try.
            # Needs to break when sum > target_prob.
            # C++: sumProbability += velDistr(v, m, temperature);
            # Note: velDistr returns probability density * roughly? No, it looks like unnormalized value.
            # Actually C++ velDistr returns a value.
            # Wait, C++ `velGenerator` accumulates `velDistr` result until it exceeds `sd`.
            # `sd` is uniform [0,1].
            # But `velDistr` is not normalized to sum to 1 in the loop?
            # Ah, `velGenerator` in C++ seems to rely on the fact that `sd` is compared to cumulative sum.
            # But if sum exceeds 1?
            # In C++, `sd` is just random number.
            # Checking `velDistr`: returns `v^5 * ... * exp(...)`.
            # This integration seems manual.
            # Let's replicate the loop.

            # Optimization: Pre-calculate velocity table?
            # C++ does it per particle? "vi = velGenerator(...)".
            # Yes.

            while sumProbability < target_prob:
                 v += dv
                 # Inline velDistr
                 v_ms = v * ANG_TO_M / FS_TO_S
                 # prob = v^5 * (m/2kT)^3 * exp(-m v^2 / 2kT)
                 # Note: This formula seems to be Maxwell-Boltzmann for flux? v^3 * v^2?
                 # Standard MB is v^2 * exp. Flux is v * f(v) ~ v^3 * exp.
                 # This uses v^5?
                 # Let's stick to C++ formula.

                 term1 = v_ms**5.0
                 term2 = (m_kg / (2.0 * BOLTZMANN_K * self.temperature_target))**3.0
                 term3 = ti.exp(-(m_kg * (v_ms**2.0)) / (2.0 * BOLTZMANN_K * self.temperature_target))

                 prob = term1 * term2 * term3
                 sumProbability += prob

            vi = v
            vcm = ti.Vector([0.0, 0.0, -vi])

            # Rotation
            gamma = 2.0 * math.pi * rnd3[i]
            phi = ti.asin(2.0 * rnd4[i] - 1.0) + 0.5 * math.pi
            theta = 2.0 * math.pi * rnd5[i]

            # Rotate rcm and vcm
            # rotate(rcm, vcm, angles=[gamma, phi, theta])
            # Rz(gamma)
            rx = rcm[0]; ry = rcm[1]; rz = rcm[2]
            vx = vcm[0]; vy = vcm[1]; vz = vcm[2]

            rcm[0] = rx * ti.cos(gamma) - ry * ti.sin(gamma)
            rcm[1] = rx * ti.sin(gamma) + ry * ti.cos(gamma)
            vcm[0] = vx * ti.cos(gamma) - vy * ti.sin(gamma)
            vcm[1] = vx * ti.sin(gamma) + vy * ti.cos(gamma)

            # Ry(phi)
            rx = rcm[0]; ry = rcm[1]; rz = rcm[2]
            vx = vcm[0]; vy = vcm[1]; vz = vcm[2]

            rcm[0] = rx * ti.cos(phi) + rz * ti.sin(phi)
            rcm[2] = rx * -ti.sin(phi) + rz * ti.cos(phi)
            vcm[0] = vx * ti.cos(phi) + vz * ti.sin(phi)
            vcm[2] = vx * -ti.sin(phi) + vz * ti.cos(phi)

            # Rz(theta)
            rx = rcm[0]; ry = rcm[1]; rz = rcm[2]
            vx = vcm[0]; vy = vcm[1]; vz = vcm[2]

            rcm[0] = rx * ti.cos(theta) - ry * ti.sin(theta)
            rcm[1] = rx * ti.sin(theta) + ry * ti.cos(theta)
            vcm[0] = vx * ti.cos(theta) - vy * ti.sin(theta)
            vcm[1] = vx * ti.sin(theta) + vy * ti.cos(theta)

            # Hit ellipsoid check
            ur = ti.Vector([rcm[0]/self.a, rcm[1]/self.b, rcm[2]/self.c])
            uv = ti.Vector([vcm[0]/self.a, vcm[1]/self.b, vcm[2]/self.c])
            cross = ur.cross(uv)
            delta = -cross.dot(cross) + uv.dot(uv)

            hit = True
            if delta < 0:
                hit = False

            if not hit:
                Nfree[i] = 1
            else:
                # Project to surface
                lambda_val = -ur.dot(uv)/uv.dot(uv) - ti.sqrt(delta)/uv.dot(uv)
                rcm += lambda_val * vcm

                # Run Dynamics (run_He logic)
                # Init
                vcm_i = vcm
                rcm_i = rcm

                dt = self.dt
                success = False
                chi = 0.0

                maxTries = 8
                trajTries = 0
                stepcount = 0

                # Initial force
                f, Up = self.force.lennardjones_LC(rcm[0], rcm[1], rcm[2])
                # Note: If force type is different (induced dipole), call correct kernel.
                # Assuming force_type 1 or 2 (LJ only) for now based on simple case.
                # If force_type 3 or 4, need induced dipole.
                # Let's add conditional logic if possible or assumes LJ for first pass.
                # The user memory says "He buffer gas... python impl currently supports He".
                # If force_type >= 3 (induced dipole), we should use `lennardjones_induced_dipole_LC`.

                if self.force_type == 3 or self.force_type == 4:
                     f, Up = self.force.lennardjones_induced_dipole_LC(rcm[0], rcm[1], rcm[2])

                fi = f

                # Kinetic Energy
                Ek = 0.5 * self.mu * vcm.dot(vcm) * amuAngfs2_to_KCAL_MOL
                Ei = Up + Ek

                rcm_old = rcm_i
                dtheta = 0.0
                theta_max = 1.5 * math.pi

                finished = False

                while trajTries < maxTries and maxTries < 10 and not finished:
                    if stepcount == 0:
                         vcm = vcm_i
                         rcm = rcm_i
                         rcm_old = rcm_i
                         f = fi
                         dtheta = 0.0

                    # First half verlet
                    vcm += 0.5 * dt / self.mu * f * KCALMOLANGAMU_TO_ANGFS2
                    rcm += dt * vcm

                    rcm_new = rcm

                    # Update angular displacement
                    # anglevec(rcm_new, rcm_old)
                    t1 = rcm_new.dot(rcm_old)
                    t2 = rcm_new.norm()
                    t3 = rcm_old.norm()
                    arg = t1 / (t2 * t3)
                    if abs(arg) <= 1.0:
                        dtheta += ti.acos(arg)
                    rcm_old = rcm_new

                    # Calculate force
                    if self.force_type == 3 or self.force_type == 4:
                        f, Up = self.force.lennardjones_induced_dipole_LC(rcm[0], rcm[1], rcm[2])
                    else:
                        f, Up = self.force.lennardjones_LC(rcm[0], rcm[1], rcm[2])

                    # Second half verlet
                    vcm += 0.5 * dt / self.mu * f * KCALMOLANGAMU_TO_ANGFS2

                    stepcount += 1

                    if dtheta > theta_max:
                        trajTries += 1
                        dt = self.dt / (trajTries + 1)
                        stepcount = 0
                        continue

                    # Check bounds
                    if abs(rcm[0]) >= self.lx or abs(rcm[1]) >= self.ly or abs(rcm[2]) >= self.lz:
                         # Check Energy conservation
                         Ek = 0.5 * self.mu * vcm.dot(vcm) * amuAngfs2_to_KCAL_MOL
                         E = Up + Ek
                         dE = abs(E - Ei) / Ei * 100.0

                         if dE > 0.5:
                             trajTries += 1
                             maxTries += 1
                             dt = self.dt / (trajTries + 1)
                             stepcount = 0
                             continue
                         else:
                             # Success
                             vcm_f = vcm
                             # chi = anglevec(vcm_i, vcm_f)
                             t1 = vcm_i.dot(vcm_f)
                             t2 = vcm_i.norm()
                             t3 = vcm_f.norm()
                             arg = t1 / (t2 * t3)
                             chi = 0.0
                             if abs(arg) <= 1.0:
                                 chi = ti.acos(arg)

                             dOmega[i] = math.pi * (1.0 - ti.cos(chi)) * (self.bmax**2.0)
                             Nscatter[i] = 1
                             finished = True

                    else:
                        # Check ellipsoid
                         ur_x = rcm[0]/self.a
                         ur_y = rcm[1]/self.b
                         ur_z = rcm[2]/self.c
                         r_ell = ur_x**2 + ur_y**2 + ur_z**2

                         if r_ell > 1.0:
                             # Check Energy
                             Ek = 0.5 * self.mu * vcm.dot(vcm) * amuAngfs2_to_KCAL_MOL
                             E = Up + Ek
                             dE = abs(E - Ei) / Ei * 100.0

                             if dE > 0.5:
                                 trajTries += 1
                                 maxTries += 1
                                 dt = self.dt / (trajTries + 1)
                                 stepcount = 0
                                 continue
                             else:
                                 # Success
                                 vcm_f = vcm
                                 t1 = vcm_i.dot(vcm_f)
                                 t2 = vcm_i.norm()
                                 t3 = vcm_f.norm()
                                 arg = t1 / (t2 * t3)
                                 chi = 0.0
                                 if abs(arg) <= 1.0:
                                     chi = ti.acos(arg)
                                 dOmega[i] = math.pi * (1.0 - ti.cos(chi)) * (self.bmax**2.0)
                                 Nscatter[i] = 1
                                 finished = True

                if not finished:
                    Nlost[i] = 1
