import pyopencl as cl
import numpy as np
from .constants import *

class GPUEngine:
    def __init__(self, molecule, params, reduced_mass, ellipsoid_params):
        self.molecule = molecule
        self.params = params
        self.mu = reduced_mass
        self.a, self.b, self.c, self.bmax = ellipsoid_params
        self.setup_opencl()

    def setup_opencl(self):
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")

        device = None
        for platform in platforms:
            for dev in platform.get_devices(device_type=cl.device_type.GPU):
                device = dev
                break
            if device: break

        if not device:
            print("Warning: No GPU found, falling back to CPU device for OpenCL")
            device = platforms[0].get_devices()[0]

        print(f"Using OpenCL device: {device.name}")
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

    def run(self):
        n_total = self.params.n_iter * self.params.n_probe

        # Buffers
        mf = cl.mem_flags

        target_x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.molecule.x)
        target_y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.molecule.y)
        target_z_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.molecule.z)
        target_eps_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.molecule.eps)
        target_sig_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.molecule.sig)

        results_chi = np.zeros(n_total, dtype=np.float64)
        results_success = np.zeros(n_total, dtype=np.int32)

        results_chi_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, results_chi.nbytes)
        results_success_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, results_success.nbytes)

        # Random numbers
        np.random.seed(self.params.seed)
        rnd1 = np.sqrt(self.bmax**2 * np.random.random(n_total)).astype(np.float64)
        rnd2 = np.random.random(n_total).astype(np.float64)
        rnd3 = np.random.random(n_total).astype(np.float64)
        rnd4 = np.random.random(n_total).astype(np.float64)
        rnd5 = np.random.random(n_total).astype(np.float64)

        rnd1_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rnd1)
        rnd2_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rnd2)
        rnd3_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rnd3)
        rnd4_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rnd4)
        rnd5_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rnd5)

        # Build Kernel
        kernel_src = self.get_kernel_source()
        prg = cl.Program(self.ctx, kernel_src).build()

        # Run
        prg.trajectory_kernel(self.queue, (n_total,), None,
                              np.int32(self.molecule.natoms),
                              target_x_buf, target_y_buf, target_z_buf,
                              target_eps_buf, target_sig_buf,
                              np.float64(self.params.dt),
                              np.float64(self.params.temperature_target),
                              np.float64(self.mu),
                              np.float64(self.a), np.float64(self.b), np.float64(self.c), np.float64(self.bmax),
                              np.float64(self.params.lj_cutoff),
                              rnd1_buf, rnd2_buf, rnd3_buf, rnd4_buf, rnd5_buf,
                              results_chi_buf, results_success_buf)

        cl.enqueue_copy(self.queue, results_chi, results_chi_buf)
        cl.enqueue_copy(self.queue, results_success, results_success_buf)

        return results_chi, results_success

    def get_kernel_source(self):
        return """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define M_PI 3.14159265358979323846
#define BOLTZMANN_K 1.38064852e-23
#define AMU_TO_KG 1.66053906660e-27
#define ANG_TO_M 1.0e-10
#define FS_TO_S 1.0e-15
#define KCALMOLANGAMU_TO_ANGFS2 4.184e-4

double velDistr(double v, double m, double temperature) {
    m = m * AMU_TO_KG;
    v = v * ANG_TO_M / FS_TO_S;
    return pow(v, 5.0) * pow((m / (2.0 * BOLTZMANN_K * temperature)), 3.0) *
      exp(-(m * pow(v, 2.0)) / (2.0 * BOLTZMANN_K * temperature));
}

double velGenerator(double m, double temperature, double sd) {
    double v = 0.0;
    double dv = 1.0E-5;
    double sumProbability = velDistr(v, m, temperature);
    int safety = 0;
    while (sumProbability < sd && safety < 100000) {
      v += dv;
      sumProbability += velDistr(v, m, temperature);
      safety++;
    }
    return v;
}

void rotate(double *r, double *v, double *angles) {
    double ri[3], vi[3];
    for(int i=0; i<3; i++) { ri[i]=r[i]; vi[i]=v[i]; }

    // Rz
    r[0] = ri[0] * cos(angles[0]) - ri[1] * sin(angles[0]);
    r[1] = ri[0] * sin(angles[0]) + ri[1] * cos(angles[0]);
    r[2] = ri[2];
    v[0] = vi[0] * cos(angles[0]) - vi[1] * sin(angles[0]);
    v[1] = vi[0] * sin(angles[0]) + vi[1] * cos(angles[0]);
    v[2] = vi[2];

    for(int i=0; i<3; i++) { ri[i]=r[i]; vi[i]=v[i]; }
    // Ry
    r[0] = ri[0] * cos(angles[1]) + ri[2] * sin(angles[1]);
    r[1] = ri[1];
    r[2] = ri[0] * -sin(angles[1]) + ri[2] * cos(angles[1]);
    v[0] = vi[0] * cos(angles[1]) + vi[2] * sin(angles[1]);
    v[1] = vi[1];
    v[2] = vi[0] * -sin(angles[1]) + vi[2] * cos(angles[1]);

    for(int i=0; i<3; i++) { ri[i]=r[i]; vi[i]=v[i]; }
    // Rz
    r[0] = ri[0] * cos(angles[2]) - ri[1] * sin(angles[2]);
    r[1] = ri[0] * sin(angles[2]) + ri[1] * cos(angles[2]);
    r[2] = ri[2];
    v[0] = vi[0] * cos(angles[2]) - vi[1] * sin(angles[2]);
    v[1] = vi[0] * sin(angles[2]) + vi[1] * cos(angles[2]);
    v[2] = vi[2];
}

double dotProduct(double *v1, double *v2) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

void crossProduct(double *v1, double *v2, double *res) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

double anglevec(double *vi, double *vf) {
    double t1 = dotProduct(vi, vf);
    double t2 = sqrt(dotProduct(vi, vi));
    double t3 = sqrt(dotProduct(vf, vf));
    double argument = t1 / (t2 * t3);
    if (fabs(argument) > 1.0) return 0.0;
    return acos(argument);
}

void lennardjones(double *r_probe, int natoms,
                  __global const double *tx, __global const double *ty, __global const double *tz,
                  __global const double *teps, __global const double *tsig,
                  double lj_cutoff,
                  double *f, double *Up) {
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double U = 0.0;
    double dx, dy, dz, r2, r2inv, r6inv, lj1, lj2, lj3, lj4, flj, epsilon, sigma;

    double rc2 = lj_cutoff*lj_cutoff;
    double rc6inv = 1.0/(rc2*rc2*rc2);

    for (int i = 0; i < natoms; i++) {
        dx = r_probe[0] - tx[i];
        dy = r_probe[1] - ty[i];
        dz = r_probe[2] - tz[i];
        r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < rc2) {
            epsilon = teps[i];
            sigma = tsig[i];

            r2inv = 1.0/r2;
            r6inv = r2inv*r2inv*r2inv;

            lj1 = 4.0*epsilon*pow(sigma, 6.0);
            lj2 = lj1*pow(sigma, 6.0);

            // Shifted Potential
            double U_term = r6inv*(lj2*r6inv - lj1);
            double U_cut = rc6inv*(lj2*rc6inv - lj1);
            U += U_term - U_cut;

            lj3 = 6.0*lj1;
            lj4 = 12.0*lj2;
            flj = r6inv*(lj4*r6inv - lj3)*r2inv;

            fx += flj*dx;
            fy += flj*dy;
            fz += flj*dz;
        }
    }
    f[0] = fx; f[1] = fy; f[2] = fz;
    *Up = U;
}

__kernel void trajectory_kernel(int natoms,
                                __global const double *tx, __global const double *ty, __global const double *tz,
                                __global const double *teps, __global const double *tsig,
                                double dt_input, double temp, double mu,
                                double a, double b, double c, double bmax,
                                double lj_cutoff,
                                __global const double *rnd1, __global const double *rnd2,
                                __global const double *rnd3, __global const double *rnd4, __global const double *rnd5,
                                __global double *results_chi, __global int *results_success) {

    int gid = get_global_id(0);

    double rcm[3], vcm[3], angles[3];
    double bi = rnd1[gid];

    rcm[0] = bi; rcm[1] = 0.0; rcm[2] = bmax;

    double vi = velGenerator(mu, temp, rnd2[gid]);
    vcm[0] = 0.0; vcm[1] = 0.0; vcm[2] = -vi;

    angles[0] = 2.0 * M_PI * rnd3[gid];
    angles[1] = asin(2.0 * rnd4[gid] - 1.0) + 0.5*M_PI;
    angles[2] = 2.0 * M_PI * rnd5[gid];

    rotate(rcm, vcm, angles);

    double ur[3], uv[3], cross[3];
    ur[0] = rcm[0]/a; ur[1] = rcm[1]/b; ur[2] = rcm[2]/c;
    uv[0] = vcm[0]/a; uv[1] = vcm[1]/b; uv[2] = vcm[2]/c;

    crossProduct(ur, uv, cross);
    double delta = -dotProduct(cross, cross) + dotProduct(uv, uv);

    if (delta < 0) {
        results_success[gid] = 0;
        return;
    }

    double lambda = -dotProduct(ur, uv)/dotProduct(uv, uv) - sqrt(delta)/dotProduct(uv, uv);

    rcm[0] += lambda*vcm[0];
    rcm[1] += lambda*vcm[1];
    rcm[2] += lambda*vcm[2];

    double rcm_i[3] = {rcm[0], rcm[1], rcm[2]};
    double vcm_i[3] = {vcm[0], vcm[1], vcm[2]};

    double f[3], Up;
    double dt = dt_input;

    lennardjones(rcm, natoms, tx, ty, tz, teps, tsig, lj_cutoff, f, &Up);

    int maxTries = 8;
    int trajTries = 0;
    int stepcount = 0;
    double rcm_old[3] = {rcm[0], rcm[1], rcm[2]};
    double dtheta = 0.0;

    while (trajTries < maxTries && maxTries < 10 && stepcount < 100000) {
        double vcm_half[3];
        for(int i=0; i<3; i++) {
            vcm_half[i] = vcm[i] + 0.5*dt/mu*f[i]*KCALMOLANGAMU_TO_ANGFS2;
            rcm[i] += dt*vcm_half[i];
        }

        dtheta += anglevec(rcm, rcm_old);
        for(int i=0; i<3; i++) rcm_old[i] = rcm[i];

        if (dtheta > 1.5 * M_PI) {
             trajTries++;
             dt = dt_input/(trajTries + 1);
             stepcount = 0;
             dtheta = 0.0;
             for(int i=0; i<3; i++) { rcm[i]=rcm_i[i]; vcm[i]=vcm_i[i]; rcm_old[i]=rcm_i[i]; }
             lennardjones(rcm, natoms, tx, ty, tz, teps, tsig, lj_cutoff, f, &Up);
             continue;
        }

        lennardjones(rcm, natoms, tx, ty, tz, teps, tsig, lj_cutoff, f, &Up);

        for(int i=0; i<3; i++) {
             vcm[i] = vcm_half[i] + 0.5*dt/mu*f[i]*KCALMOLANGAMU_TO_ANGFS2;
        }

        stepcount++;

        double ur_curr[3];
        ur_curr[0] = rcm[0]/a; ur_curr[1] = rcm[1]/b; ur_curr[2] = rcm[2]/c;
        if (dotProduct(ur_curr, ur_curr) > 1.0) {
             double chi = anglevec(vcm_i, vcm);
             results_chi[gid] = chi;
             results_success[gid] = 1;
             return;
        }
    }
    results_success[gid] = -1;
}
"""
