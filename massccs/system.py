import numpy as np
import math
import sys
import time

from .molecule_target import MoleculeTarget
from .gas_buffer import GasBuffer
from .constants import *

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    print("PyOpenCL not found. Falling back to CPU (which might be slow).")

class System:
    def __init__(self, target_filename="c60.xyz", n_probe=10000, n_iter=10,
                 dt=10.0, temp=298.0, gas_type="N2", use_cpu=False):
        self.target_filename = target_filename
        self.n_probe = n_probe
        self.n_iter = n_iter
        self.dt = dt
        self.temp = temp
        self.gas_type = gas_type # Only N2 implemented
        self.use_cpu = use_cpu or (not HAS_OPENCL)

        # Load Target
        self.target = MoleculeTarget(target_filename)

        # Load Gas Properties
        self.gas = GasBuffer(gas_flag=2) # N2

        # Ellipsoid Parameters
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.geometric_ellipsoid()

        self.bmax = max(self.a, self.b, self.c)

        # Linked Cell (Not implemented as requested "no cutoffs")
        self.lx = self.bmax * 2.5 # Simulation box guess
        self.ly = self.bmax * 2.5
        self.lz = self.bmax * 2.5

        # Reduced Mass
        # N2: flag 2
        self.mu = self.gas.mass
        self.Inertia = 0.0
        for i in range(self.gas.natoms):
            self.Inertia += self.gas.m[i] * (self.gas.z[i]**2)

        # OpenCL Setup
        if HAS_OPENCL and not self.use_cpu:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            with open('massccs/kernels.cl', 'r') as f:
                self.prg = cl.Program(self.ctx, f.read()).build()
        else:
            self.ctx = None

    def geometric_ellipsoid(self):
        # Ported from System.cpp
        # Using simplified LJ radius as radius
        # Default LJ cutoff 12.0
        radius = 2.0 * 12.0
        skin = 0.01
        d = self.gas.d

        self.a = self.target.maxX + radius + skin + d
        self.b = self.target.maxY + radius + skin + d
        self.c = self.target.maxZ + radius + skin + d

        delta = 0.01

        # Expand until all atoms fit?
        # C++ code expands a,b,c until all atoms + radius fit inside x^2/a^2 + ... <= 1

        outside = True
        while outside:
            outside = False
            for i in range(self.target.natoms):
                xi = self.target.x[i]
                yi = self.target.y[i]
                zi = self.target.z[i]
                ri = math.sqrt(xi*xi + yi*yi + zi*zi)
                if ri > 1e-9:
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

    def vel_generator(self, m, temp):
        # Gamma sampling for velocity
        m_kg = m * AMU_TO_KG
        k = BOLTZMANN_K
        a = m_kg / (2.0 * k * temp)
        x = np.random.gamma(3.0, 1.0/a)
        v_ms = math.sqrt(x)
        return v_ms * (ANG_TO_M / FS_TO_S)**-1

    def run(self):
        print(f"Starting simulation: N2 on {self.target_filename}")
        print(f"Target Mass: {self.target.mass} amu")
        print(f"Ellipsoid: {self.a:.2f}, {self.b:.2f}, {self.c:.2f} Ang")
        print(f"Max Impact Parameter: {self.bmax:.2f} Ang")
        print(f"Backend: {'OpenCL' if (HAS_OPENCL and not self.use_cpu) else 'CPU'}")

        # Accumulators
        Omega = 0.0
        Omega2 = 0.0

        # Loop n_iter
        for it in range(self.n_iter):
            # Generate initial conditions for n_probe trajectories

            # 1. Impact parameter b ~ sqrt(rnd) * bmax
            rnd1 = np.random.random(self.n_probe)
            bi = np.sqrt(rnd1) * self.bmax

            # 2. Velocity
            m_kg = self.mu * AMU_TO_KG
            k = BOLTZMANN_K
            scale = (2.0 * k * self.temp) / m_kg
            x_gamma = np.random.gamma(3.0, scale, self.n_probe)
            vi_mag = np.sqrt(x_gamma) * (ANG_TO_M / FS_TO_S)**-1

            # 3. Orientations
            gamma = 2.0 * np.pi * np.random.random(self.n_probe)
            phi = np.arcsin(2.0 * np.random.random(self.n_probe) - 1.0) + 0.5*np.pi
            theta = 2.0 * np.pi * np.random.random(self.n_probe)

            # N2 internal rotation
            theta_n2 = np.arcsin(2.0 * np.random.random(self.n_probe) - 1.0) + 0.5*np.pi
            phi_n2 = 2.0 * np.pi * np.random.random(self.n_probe)
            psi_n2 = 2.0 * np.pi * np.random.random(self.n_probe)

            rnd_w = np.random.random(self.n_probe)
            omega_n2 = np.sqrt((2.0 * k * self.temp / (self.Inertia*AMU_TO_KG*ANG_TO_M**2)) * np.log(1.0/(1.0-rnd_w))) * (FS_TO_S) # rad/fs

            # Setup Initial State Arrays (Host)
            initial_pos = np.zeros((self.n_probe, 9), dtype=np.float64)
            initial_vel = np.zeros((self.n_probe, 9), dtype=np.float64)

            rcm = np.zeros((self.n_probe, 3))
            rcm[:, 0] = bi
            rcm[:, 2] = self.bmax

            vcm = np.zeros((self.n_probe, 3))
            vcm[:, 2] = -vi_mag

            def vec_rotate_z(r, v, angle):
                c = np.cos(angle)
                s = np.sin(angle)
                rx = r[:,0]*c - r[:,1]*s
                ry = r[:,0]*s + r[:,1]*c
                r[:,0] = rx; r[:,1] = ry

                vx = v[:,0]*c - v[:,1]*s
                vy = v[:,0]*s + v[:,1]*c
                v[:,0] = vx; v[:,1] = vy

            def vec_rotate_y(r, v, angle):
                c = np.cos(angle)
                s = np.sin(angle)
                rx = r[:,0]*c + r[:,2]*s
                rz = -r[:,0]*s + r[:,2]*c
                r[:,0] = rx; r[:,2] = rz

                vx = v[:,0]*c + v[:,2]*s
                vz = -v[:,0]*s + v[:,2]*c
                v[:,0] = vx; v[:,2] = vz

            vec_rotate_z(rcm, vcm, gamma)
            vec_rotate_y(rcm, vcm, phi)
            vec_rotate_z(rcm, vcm, theta)

            # Project to Ellipsoid Surface
            ur = rcm / np.array([self.a, self.b, self.c])
            uv = vcm / np.array([self.a, self.b, self.c])

            uv_sq = np.sum(uv**2, axis=1)
            ur_uv = np.sum(ur*uv, axis=1)
            ur_sq = np.sum(ur**2, axis=1)

            D_4 = ur_uv**2 - uv_sq * ur_sq + uv_sq
            hit_mask = D_4 >= 0

            lamb = (-ur_uv - np.sqrt(np.maximum(0, D_4))) / uv_sq
            rcm[hit_mask] += lamb[hit_mask, None] * vcm[hit_mask]

            # Setup N2 atoms
            d_half = self.gas.d / 2.0
            r_gas = np.zeros((self.n_probe, 3, 3))
            r_gas[:, 0, 2] = d_half
            r_gas[:, 2, 2] = -d_half

            # Rotate gas
            def vec_rotate_gas(rg, theta, phi):
                c = np.cos(theta); s = np.sin(theta)
                c = c[:, None]; s = s[:, None]
                rx = rg[:, :, 0]*c + rg[:, :, 2]*s
                rz = -rg[:, :, 0]*s + rg[:, :, 2]*c
                rg[:, :, 0] = rx; rg[:, :, 2] = rz

                c = np.cos(phi); s = np.sin(phi)
                c = c[:, None]; s = s[:, None]
                rx = rg[:, :, 0]*c - rg[:, :, 1]*s
                ry = rg[:, :, 0]*s + rg[:, :, 1]*c
                rg[:, :, 0] = rx; rg[:, :, 1] = ry

            vec_rotate_gas(r_gas, theta_n2, phi_n2)

            wx = omega_n2 * np.cos(psi_n2)
            wy = omega_n2 * np.sin(psi_n2)
            wz = np.zeros(self.n_probe)
            w_gas = np.stack([wx, wy, wz], axis=1) # [N, 3]

            # Rotate w_gas
            c = np.cos(theta_n2); s = np.sin(theta_n2)
            wx_new = w_gas[:,0]*c + w_gas[:,2]*s
            wz_new = -w_gas[:,0]*s + w_gas[:,2]*c
            w_gas[:,0] = wx_new; w_gas[:,2] = wz_new

            c = np.cos(phi_n2); s = np.sin(phi_n2)
            wx_new = w_gas[:,0]*c - w_gas[:,1]*s
            wy_new = w_gas[:,0]*s + w_gas[:,1]*c
            w_gas[:,0] = wx_new; w_gas[:,1] = wy_new

            v_gas = np.cross(w_gas[:, None, :], r_gas)

            r_gas += rcm[:, None, :]
            v_gas += vcm[:, None, :]

            initial_pos[:, 0:3] = r_gas[:, 0, :]
            initial_pos[:, 3:6] = r_gas[:, 1, :]
            initial_pos[:, 6:9] = r_gas[:, 2, :]

            initial_vel[:, 0:3] = v_gas[:, 0, :]
            initial_vel[:, 3:6] = v_gas[:, 1, :]
            initial_vel[:, 6:9] = v_gas[:, 2, :]

            results = np.zeros(self.n_probe, dtype=np.int32)
            chi_results = np.zeros(self.n_probe, dtype=np.float64)
            idxs = np.where(hit_mask)[0]
            n_hits = len(idxs)

            if n_hits > 0:
                if HAS_OPENCL and not self.use_cpu:
                    mf = cl.mem_flags
                    target_dtype = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                                             ('eps', 'f8'), ('sig', 'f8'), ('q', 'f8')])
                    target_data = np.zeros(self.target.natoms, dtype=target_dtype)
                    target_data['x'] = self.target.x
                    target_data['y'] = self.target.y
                    target_data['z'] = self.target.z
                    target_data['eps'] = self.target.eps
                    target_data['sig'] = self.target.sig
                    target_data['q'] = self.target.q

                    target_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_data)

                    hit_pos = np.ascontiguousarray(initial_pos[idxs])
                    hit_vel = np.ascontiguousarray(initial_vel[idxs])

                    pos_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hit_pos)
                    vel_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hit_vel)
                    res_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=n_hits * 4)
                    chi_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=n_hits * 8)

                    d2 = self.gas.d**2
                    max_steps = 100000
                    lj_cutoff_sq = 12.0**2

                    kernel = cl.Kernel(self.prg, 'run_N2_trajectories')
                    kernel(
                        self.queue, (n_hits,), None,
                        target_buf, np.int32(self.target.natoms),
                        pos_buf, vel_buf, res_buf, chi_buf,
                        np.float64(self.dt),
                        np.float64(self.a), np.float64(self.b), np.float64(self.c),
                        np.float64(self.lx), np.float64(self.ly), np.float64(self.lz),
                        np.float64(d2), np.int32(max_steps),
                        np.float64(lj_cutoff_sq)
                    )

                    hit_results = np.zeros(n_hits, dtype=np.int32)
                    hit_chi = np.zeros(n_hits, dtype=np.float64)

                    cl.enqueue_copy(self.queue, hit_results, res_buf)
                    cl.enqueue_copy(self.queue, hit_chi, chi_buf)

                    results[idxs] = hit_results
                    chi_results[idxs] = hit_chi
                else:
                    # CPU Implementation
                    print("Running on CPU (this may be slow)...")

                    hit_pos = initial_pos[idxs]
                    hit_vel = initial_vel[idxs]
                    hit_results = np.zeros(n_hits, dtype=np.int32)
                    hit_chi = np.zeros(n_hits, dtype=np.float64)

                    d2 = self.gas.d**2
                    max_steps = 100000
                    lj_cutoff_sq = 12.0**2
                    dt = self.dt

                    for i in range(n_hits):
                        # Extract single probe
                        r = hit_pos[i].copy()
                        v = hit_vel[i].copy()

                        m_N = 14.007

                        # Calculate initial energy for conservation check
                        Ek_i = 0.5 * m_N * (np.sum(v[0:3]**2) + np.sum(v[6:9]**2)) * 2390.07

                        # Initial Forces
                        def calc_force(r_atom, target):
                            f_acc = np.zeros(3)
                            U_acc = 0.0

                            # Vectorized distance to all targets
                            delta = r_atom - np.stack([target.x, target.y, target.z], axis=1)
                            r2 = np.sum(delta**2, axis=1)

                            mask = r2 < lj_cutoff_sq
                            if not np.any(mask):
                                return f_acc, U_acc

                            r2 = r2[mask]
                            delta = delta[mask]
                            eps = target.eps[mask]
                            sig = target.sig[mask]

                            r2inv = 1.0 / r2
                            r6inv = r2inv**3

                            sig2 = sig**2
                            sig6 = sig2**3

                            lj1 = 4.0 * eps * sig6
                            lj2 = lj1 * sig6

                            Ulj = r6inv * (lj2 * r6inv - lj1)

                            rc2inv = 1.0 / lj_cutoff_sq
                            rc6inv = rc2inv**3
                            Ulj_cut = rc6inv * (lj2 * rc6inv - lj1)

                            U_acc = np.sum(Ulj - Ulj_cut)

                            flj = r6inv * (12.0 * lj2 * r6inv - 6.0 * lj1) * r2inv
                            f_acc = np.sum(flj[:, None] * delta, axis=0)

                            return f_acc, U_acc

                        # Dummy target object wrapper or passing self.target
                        f0, U0 = calc_force(r[0:3], self.target)
                        f2, U2 = calc_force(r[6:9], self.target)

                        Up_total = U0 + U2
                        Ei = Ek_i + Up_total

                        # VCM initial
                        vcm_i = 0.5 * (v[0:3] + v[6:9])

                        step = 0
                        status = 2 # Default Lost
                        chi = 0.0

                        fi = f0
                        fj = f2

                        # Pre-calc constants
                        muij = 0.5 * m_N # m*m/(2m) = m/2
                        dt_m = 0.5 * dt / m_N * KCALMOLANGAMU_TO_ANGFS2

                        while step < max_steps:
                            # First Half Verlet Constrained
                            ri = r[0:3]; rj = r[6:9]
                            vi = v[0:3]; vj = v[6:9]

                            vci = vi + fi * dt_m
                            vcj = vj + fj * dt_m

                            rci = ri + dt * vci
                            rcj = rj + dt * vcj

                            rij = ri - rj
                            rcij = rci - rcj

                            r2ij = np.dot(rij, rij)
                            rc2ij = np.dot(rcij, rcij)
                            rcijrij = np.dot(rcij, rij)

                            determ = rcijrij**2 - r2ij * (rc2ij - d2)
                            if determ < 0: determ = 0.0

                            lam = (rcijrij - math.sqrt(determ)) / r2ij

                            # Update full step
                            # muij / (mi * dt) = (m/2) / (m * dt) = 1 / (2dt)
                            # muij / mi = 0.5

                            factor_v = 1.0 / (2.0 * dt) * lam
                            factor_r = 0.5 * lam

                            vi = vci - factor_v * rij
                            vj = vcj + factor_v * rij

                            ri = rci - factor_r * rij
                            rj = rcj + factor_r * rij

                            # Calc Dummy (COM)
                            rcm_curr = 0.5 * (ri + rj)
                            vcm_curr = 0.5 * (vi + vj)

                            # Forces
                            f0, U0 = calc_force(ri, self.target)
                            f2, U2 = calc_force(rj, self.target)

                            fi_next = f0
                            fj_next = f2
                            Up_total = U0 + U2

                            # Second Half Verlet
                            # vci = vi + fi_next * dt_m
                            # vcj = vj + fj_next * dt_m
                            # rij = ri - rj
                            # vcij = (vi + fi_next * dt_m) - (vj + fj_next * dt_m)
                            # lambda = mu * rij.vcij / r2ij

                            vci = vi + fi_next * dt_m
                            vcj = vj + fj_next * dt_m

                            rij_new = ri - rj
                            vcij = vci - vcj

                            rijvcij = np.dot(rij_new, vcij)
                            r2ij_new = np.dot(rij_new, rij_new)

                            lam2 = muij * rijvcij / r2ij_new

                            vi = vci - (lam2 / m_N) * rij_new
                            vj = vcj + (lam2 / m_N) * rij_new

                            # Update local vars
                            r[0:3] = ri; r[6:9] = rj
                            v[0:3] = vi; v[6:9] = vj

                            fi = fi_next
                            fj = fj_next

                            step += 1

                            # Check bounds
                            rcm_x, rcm_y, rcm_z = rcm_curr

                            # Outside box
                            if abs(rcm_x) >= self.lx or abs(rcm_y) >= self.ly or abs(rcm_z) >= self.lz:
                                Ek = 0.5 * m_N * (np.sum(vi**2) + np.sum(vj**2)) * 2390.07
                                Ef = Ek + Up_total
                                if abs((Ef - Ei)/Ei) > 0.05: # Relaxed tolerance for CPU/Python?
                                    status = 2
                                else:
                                    status = 1
                                    # Calc Chi
                                    vcm_f = 0.5 * (vi + vj)
                                    dot = np.dot(vcm_i, vcm_f)
                                    mod_i = np.linalg.norm(vcm_i)
                                    mod_f = np.linalg.norm(vcm_f)
                                    arg = dot / (mod_i * mod_f)
                                    arg = max(-1.0, min(1.0, arg))
                                    chi = math.acos(arg)
                                break

                            # Outside Ellipsoid
                            ur_val = (rcm_x/self.a)**2 + (rcm_y/self.b)**2 + (rcm_z/self.c)**2
                            if ur_val > 1.0:
                                Ek = 0.5 * m_N * (np.sum(vi**2) + np.sum(vj**2)) * 2390.07
                                Ef = Ek + Up_total
                                if abs((Ef - Ei)/Ei) > 0.05:
                                    status = 2
                                else:
                                    status = 1
                                    vcm_f = 0.5 * (vi + vj)
                                    dot = np.dot(vcm_i, vcm_f)
                                    mod_i = np.linalg.norm(vcm_i)
                                    mod_f = np.linalg.norm(vcm_f)
                                    arg = dot / (mod_i * mod_f)
                                    arg = max(-1.0, min(1.0, arg))
                                    chi = math.acos(arg)
                                break

                        hit_results[i] = status
                        hit_chi[i] = chi

                    results[idxs] = hit_results
                    chi_results[idxs] = hit_chi

                scatter_mask = (results == 1)
                lost_mask = (results == 2)
                free_mask = (results == 0) # non-hits

                dOmega_sum = np.sum(np.pi * (1.0 - np.cos(chi_results[scatter_mask])) * (self.bmax**2))

                Nscatter = np.sum(scatter_mask)
                Nlost = np.sum(lost_mask)
                Nfree = np.sum(free_mask)

                total_valid = Nscatter + Nfree
                if total_valid > 0:
                    ccs = dOmega_sum / total_valid
                else:
                    ccs = 0.0

                Omega += ccs
                Omega2 += ccs**2

                print(f"Iter {it+1}: CCS={ccs:.4f}, Scatter={Nscatter}, Free={Nfree}, Lost={Nlost}")

            else:
                print("No hits generated.")

        # Final Average
        CCS_ave = Omega / self.n_iter
        if self.n_iter > 1:
            sig2 = Omega2 / self.n_iter - CCS_ave**2
            CCS_err = math.sqrt(sig2 / self.n_iter)
        else:
            CCS_err = 0.0

        print("*********************************************************")
        print(f"average value of CCS = {CCS_ave:.4f} Ang^2")
        print(f"error value of CCS = {CCS_err:.4f} Ang^2")
        print("*********************************************************")
