import numpy as np
import time
from .constants import *
from .core import velGenerator_host

class CPUEngine:
    def __init__(self, molecule, params, reduced_mass, ellipsoid_params):
        self.molecule = molecule
        self.params = params
        self.mu = reduced_mass
        self.a, self.b, self.c, self.bmax = ellipsoid_params

    def run(self):
        print("Starting CPU Simulation...")
        start_time = time.time()

        n_total = self.params.n_iter * self.params.n_probe

        # Pre-generate random numbers
        np.random.seed(self.params.seed)
        rnd1 = np.random.random(n_total) # for b
        rnd2 = np.random.random(n_total) # for v
        rnd3 = np.random.random(n_total) # gamma
        rnd4 = np.random.random(n_total) # phi
        rnd5 = np.random.random(n_total) # theta

        # Calculate impact parameters
        b_vec = np.sqrt(self.bmax**2 * rnd1)

        # Calculate velocities
        # This is the slow part on CPU if not vectorized
        print("Generating velocities...")
        v_vec = velGenerator_host(self.mu, self.params.temperature_target, rnd2)

        print("Running trajectories...")

        results_chi = []
        results_success = []

        # We can parallelize this or run in a loop
        # For verification, a loop is fine, maybe print every 100

        for i in range(n_total):
            chi, success = self.run_trajectory(b_vec[i], v_vec[i], rnd3[i], rnd4[i], rnd5[i])
            results_chi.append(chi)
            results_success.append(success)

            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{n_total}")

        end_time = time.time()
        print(f"CPU Simulation finished in {end_time - start_time:.2f}s")

        return np.array(results_chi), np.array(results_success)

    def run_trajectory(self, b, v, rnd_gamma, rnd_phi, rnd_theta):
        # 1. Setup
        rcm = np.array([b, 0.0, self.bmax], dtype=np.float64)
        vcm = np.array([0.0, 0.0, -v], dtype=np.float64)

        # Angles
        gamma = 2.0 * np.pi * rnd_gamma
        phi = np.arcsin(2.0 * rnd_phi - 1.0) + 0.5 * np.pi
        theta = 2.0 * np.pi * rnd_theta
        angles = np.array([gamma, phi, theta])

        # Rotate
        rcm, vcm = self.rotate(rcm, vcm, angles)

        # Project to ellipsoid surface
        ur = rcm / np.array([self.a, self.b, self.c])
        uv = vcm / np.array([self.a, self.b, self.c])

        cross = np.cross(ur, uv)
        delta = -np.dot(cross, cross) + np.dot(uv, uv)

        if delta < 0:
            return 0.0, 0 # Free/Miss

        lambda_val = -np.dot(ur, uv)/np.dot(uv, uv) - np.sqrt(delta)/np.dot(uv, uv)

        rcm += lambda_val * vcm

        # 2. Integration
        rcm_i = rcm.copy()
        vcm_i = vcm.copy()
        rcm_old = rcm.copy()

        dt = self.params.dt

        # Initial Force
        f, Up = self.calculate_force(rcm)

        maxTries = 8
        trajTries = 0
        stepcount = 0
        dtheta = 0.0

        while trajTries < maxTries and maxTries < 10 and stepcount < 100000:
            # First half Verlet
            # v(t+dt/2) = v(t) + 0.5 * a(t) * dt
            # r(t+dt) = r(t) + v(t+dt/2) * dt

            # Acceleration in A/fs^2
            # f is in kcal/mol/A. mu in AMU.
            # accel = f / mu * CONST
            accel = (f / self.mu) * KCALMOLANGAMU_TO_ANGFS2

            vcm_half = vcm + 0.5 * dt * accel
            rcm = rcm + dt * vcm_half

            # Angular displacement check
            # angle between rcm and rcm_old
            # cos(theta) = dot / (|a||b|)

            # Handling small steps where rcm ~ rcm_old
            norm_r = np.linalg.norm(rcm)
            norm_r_old = np.linalg.norm(rcm_old)
            if norm_r > 1e-12 and norm_r_old > 1e-12:
                dot = np.dot(rcm, rcm_old)
                val = dot / (norm_r * norm_r_old)
                if val > 1.0: val = 1.0
                if val < -1.0: val = -1.0
                dtheta += np.arccos(val)

            rcm_old = rcm.copy()

            if dtheta > 1.5 * np.pi:
                # Reduce dt and restart
                trajTries += 1
                dt = self.params.dt / (trajTries + 1)
                stepcount = 0
                dtheta = 0.0
                rcm = rcm_i.copy()
                vcm = vcm_i.copy()
                rcm_old = rcm_i.copy()
                f, Up = self.calculate_force(rcm)
                continue

            # Calculate new Force
            f_new, Up_new = self.calculate_force(rcm)

            # Second half Verlet
            # v(t+dt) = v(t+dt/2) + 0.5 * a(t+dt) * dt
            accel_new = (f_new / self.mu) * KCALMOLANGAMU_TO_ANGFS2
            vcm = vcm_half + 0.5 * dt * accel_new

            f = f_new # update force for next step? No, Verlet needs F(t+dt) which we just calc.
            # Actually standard Velocity Verlet:
            # 1. v(t+0.5dt) = v(t) + 0.5*a(t)*dt
            # 2. r(t+dt) = r(t) + v(t+0.5dt)*dt
            # 3. a(t+dt) = F(r(t+dt))/m
            # 4. v(t+dt) = v(t+0.5dt) + 0.5*a(t+dt)*dt
            # My code matches this.

            stepcount += 1

            # Check bounds (Ellipsoid)
            ur_curr = rcm / np.array([self.a, self.b, self.c])
            if np.dot(ur_curr, ur_curr) > 1.0:
                # Outside
                # Calculate scattering angle

                # Check energy conservation? (Skipped for now for speed/simplicity, C++ does it)

                # Calculate chi
                norm_vi = np.linalg.norm(vcm_i)
                norm_vf = np.linalg.norm(vcm)
                dot = np.dot(vcm_i, vcm)
                val = dot / (norm_vi * norm_vf)
                if val > 1.0: val = 1.0
                if val < -1.0: val = -1.0
                chi = np.arccos(val)

                return chi, 1 # Success

        return 0.0, -1 # Lost/Trapped

    def calculate_force(self, r_probe):
        # Lennard-Jones Force
        # F = sum( F_i )
        # F_i = 24 eps ( 2(sig/r)^12 - (sig/r)^6 ) * r_vec / r^2
        # My derivation:
        # U = 4 eps ( (sig/r)^12 - (sig/r)^6 )
        # F = -dU/dr = -4 eps ( 12(sig/r)^11 * (-sig/r^2) - 6(sig/r)^5 * (-sig/r^2) )
        #   = 4 eps ( 12 sig^12 / r^13 - 6 sig^6 / r^7 )
        #   = 24 eps / r * ( 2 (sig/r)^12 - (sig/r)^6 )
        # F_vec = F * (r_vec / r) = F/r * r_vec
        #       = 24 eps / r^2 * ( 2 (sig/r)^12 - (sig/r)^6 ) * r_vec

        # C++ Code uses:
        # r6inv = (1/r2)^3
        # lj1 = 4*eps*sig^6
        # lj2 = lj1*sig^6 = 4*eps*sig^12
        # flj = r6inv*(12*lj2*r6inv - 6*lj1)*r2inv
        #     = r^-6 * (48 eps sig^12 r^-6 - 24 eps sig^6) * r^-2
        #     = (48 eps sig^12 r^-12 - 24 eps sig^6 r^-6) * r^-2
        #     = (48 eps sig^12 / r^14 - 24 eps sig^6 / r^8)
        # Force_x = flj * dx
        # F = flj * r
        #   = 48 eps sig^12 / r^13 - 24 eps sig^6 / r^7
        #   = 24 eps / r ( 2 sig^12/r^12 - sig^6/r^6 ) matches.

        # Using numpy broadcasting
        dx = r_probe[0] - self.molecule.x
        dy = r_probe[1] - self.molecule.y
        dz = r_probe[2] - self.molecule.z

        r2 = dx*dx + dy*dy + dz*dz
        r2inv = 1.0 / r2
        r6inv = r2inv * r2inv * r2inv

        # Cutoff check
        # We can filter indices where r < cutoff
        mask = np.sqrt(r2) < self.params.lj_cutoff

        if not np.any(mask):
            return np.array([0.0, 0.0, 0.0]), 0.0

        r2inv = r2inv[mask]
        r6inv = r6inv[mask]

        eps = self.molecule.eps[mask]
        sig = self.molecule.sig[mask]

        lj1 = 4.0 * eps * (sig**6)
        lj2 = lj1 * (sig**6)

        # Potential
        U_terms = r6inv * (lj2 * r6inv - lj1)
        # Subtract cutoff potential? (Shifted LJ)
        # C++: Ulj_cut = rc6inv*(lj2*rc6inv - lj1);
        # U += Ulj - Ulj_cut
        # We need rc6inv
        rc = self.params.lj_cutoff
        rc2 = rc*rc
        rc6inv = 1.0 / (rc2*rc2*rc2)
        U_cut = rc6inv * (lj2 * rc6inv - lj1)

        Up = np.sum(U_terms - U_cut)

        # Force
        lj3 = 6.0 * lj1
        lj4 = 12.0 * lj2

        flj = r6inv * (lj4 * r6inv - lj3) * r2inv

        fx = np.sum(flj * dx[mask])
        fy = np.sum(flj * dy[mask])
        fz = np.sum(flj * dz[mask])

        return np.array([fx, fy, fz]), Up

    def rotate(self, r, v, angles):
        # Rotations Rz(gamma) Ry(phi) Rz(theta)

        # Rz(angle[0])
        c, s = np.cos(angles[0]), np.sin(angles[0])
        r_new = r.copy(); v_new = v.copy()
        r_new[0] = r[0]*c - r[1]*s
        r_new[1] = r[0]*s + r[1]*c
        v_new[0] = v[0]*c - v[1]*s
        v_new[1] = v[0]*s + v[1]*c
        r = r_new; v = v_new

        # Ry(angle[1])
        c, s = np.cos(angles[1]), np.sin(angles[1])
        r_new = r.copy(); v_new = v.copy()
        # y is unchanged
        # x' = x cos + z sin
        # z' = -x sin + z cos
        r_new[0] = r[0]*c + r[2]*s
        r_new[2] = -r[0]*s + r[2]*c
        v_new[0] = v[0]*c + v[2]*s
        v_new[2] = -v[0]*s + v[2]*c
        r = r_new; v = v_new

        # Rz(angle[2])
        c, s = np.cos(angles[2]), np.sin(angles[2])
        r_new = r.copy(); v_new = v.copy()
        r_new[0] = r[0]*c - r[1]*s
        r_new[1] = r[0]*s + r[1]*c
        v_new[0] = v[0]*c - v[1]*s
        v_new[1] = v[0]*s + v[1]*c
        r = r_new; v = v_new

        return r, v
