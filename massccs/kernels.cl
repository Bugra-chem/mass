// OpenCL Kernels for MassCCS N2 Simulation
// This file contains the kernel for simulating N2 trajectories around a C60 molecule.

#define KCALMOLANGAMU_TO_ANGFS2 (1.0/48.88821291/48.88821291)
#define M_PI 3.14159265358979323846

// Structure for Target Atom
typedef struct {
    double x, y, z;
    double eps, sig;
    double q;
} TargetAtom;

// Function to calculate LJ force between a probe atom and all target atoms
// Returns potential energy U and updates force f
void lennardjones(double rx, double ry, double rz,
                  __constant TargetAtom* target, int n_target,
                  double* fx, double* fy, double* fz, double* U,
                  double lj_cutoff_sq)
{
    double dx, dy, dz, r2, r6inv, lj1, lj2;
    double flj, fx_acc=0.0, fy_acc=0.0, fz_acc=0.0, U_acc=0.0;
    double r2inv, rc2inv, rc6inv, Ulj_cut;

    // Precompute cutoff shift terms?
    // Shift depends on eps/sig which vary per atom.
    // So must compute inside loop.

    for (int i=0; i<n_target; i++) {
        dx = rx - target[i].x;
        dy = ry - target[i].y;
        dz = rz - target[i].z;
        r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < lj_cutoff_sq) {
            double epsilon = target[i].eps;
            double sigma = target[i].sig;
            double sig2 = sigma*sigma;
            double sig6 = sig2*sig2*sig2;

            lj1 = 4.0 * epsilon * sig6;
            lj2 = lj1 * sig6;

            r2inv = 1.0/r2;
            r6inv = r2inv*r2inv*r2inv;

            double Ulj = r6inv * (lj2 * r6inv - lj1);

            // Shift
            // rc6inv = 1.0 / cutoff^6
            rc2inv = 1.0 / lj_cutoff_sq;
            rc6inv = rc2inv * rc2inv * rc2inv;
            Ulj_cut = rc6inv * (lj2 * rc6inv - lj1);

            U_acc += Ulj - Ulj_cut;

            flj = r6inv * (12.0 * lj2 * r6inv - 6.0 * lj1) * r2inv;

            fx_acc += flj * dx;
            fy_acc += flj * dy;
            fz_acc += flj * dz;
        }
    }

    *fx = fx_acc;
    *fy = fy_acc;
    *fz = fz_acc;
    *U = U_acc;
}

// Constraint solver (First Half)
void first_half_verlet_constrained(double* ri, double* rj, double* vi, double* vj,
                                   double* fi, double* fj, double mi, double mj, double dt, double d2ij)
{
    double rci[3], rcj[3], vci[3], vcj[3], rcij[3], rij[3];
    double muij = mi*mj/(mi+mj);
    double dt_mi = 0.5*dt/mi * KCALMOLANGAMU_TO_ANGFS2;
    double dt_mj = 0.5*dt/mj * KCALMOLANGAMU_TO_ANGFS2;

    for(int k=0; k<3; k++) {
        vci[k] = vi[k] + fi[k] * dt_mi;
        vcj[k] = vj[k] + fj[k] * dt_mj;
        rci[k] = ri[k] + dt * vci[k];
        rcj[k] = rj[k] + dt * vcj[k];
        rij[k] = ri[k] - rj[k];
        rcij[k] = rci[k] - rcj[k];
    }

    double r2ij = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
    double rc2ij = rcij[0]*rcij[0] + rcij[1]*rcij[1] + rcij[2]*rcij[2];
    double rcijrij = rcij[0]*rij[0] + rcij[1]*rij[1] + rcij[2]*rij[2];

    double determ = rcijrij*rcijrij - r2ij*(rc2ij - d2ij);
    if (determ < 0.0) determ = 0.0;

    double lambda = (rcijrij - sqrt(determ))/r2ij;

    for(int k=0; k<3; k++) {
        vi[k] = vci[k] - muij/(mi*dt)*lambda*rij[k];
        vj[k] = vcj[k] + muij/(mj*dt)*lambda*rij[k];
        ri[k] = rci[k] - muij/mi*lambda*rij[k];
        rj[k] = rcj[k] + muij/mi*lambda*rij[k];
    }
}

// Constraint solver (Second Half)
void second_half_verlet_constrained(double* ri, double* rj, double* vi, double* vj,
                                    double* fi, double* fj, double mi, double mj, double dt)
{
    double vci[3], vcj[3], rij[3], vcij[3];
    double muij = mi*mj/(mi+mj);
    double dt_mi = 0.5*dt/mi * KCALMOLANGAMU_TO_ANGFS2;
    double dt_mj = 0.5*dt/mj * KCALMOLANGAMU_TO_ANGFS2;

    for(int k=0; k<3; k++) {
        vci[k] = vi[k] + fi[k] * dt_mi;
        vcj[k] = vj[k] + fj[k] * dt_mj;
        rij[k] = ri[k] - rj[k];
        vcij[k] = vci[k] - vcj[k];
    }

    double rijvcij = rij[0]*vcij[0] + rij[1]*vcij[1] + rij[2]*vcij[2];
    double r2ij = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];

    double lambda = muij * rijvcij / r2ij;

    for(int k=0; k<3; k++) {
        vi[k] = vci[k] - lambda/mi * rij[k];
        vj[k] = vcj[k] + lambda/mj * rij[k];
    }
}

// Helper to calculate Kinetic Energy
double kinetic_energy(double m, double vx, double vy, double vz) {
    return 0.5 * m * (vx*vx + vy*vy + vz*vz) * 2390.07;
}

// Main Kernel
__kernel void run_N2_trajectories(
    __constant TargetAtom* target,
    int n_target,
    __global double* initial_pos, // [Ntraj * 9] (3 atoms * 3 coords)
    __global double* initial_vel, // [Ntraj * 9]
    __global int* results, // [Ntraj]: 0=Free, 1=Scatter, 2=Lost
    __global double* chi_results, // [Ntraj]
    double dt,
    double a, double b, double c, // Ellipsoid parameters
    double lx, double ly, double lz, // Box parameters
    double d_bond_sq,
    int max_steps,
    double lj_cutoff_sq
)
{
    int gid = get_global_id(0);

    // Load initial state
    double r[9]; // x0, y0, z0, x1, y1, z1, x2, y2, z2
    double v[9];

    for(int k=0; k<9; k++) {
        r[k] = initial_pos[gid*9 + k];
        v[k] = initial_vel[gid*9 + k];
    }

    // N2 Parameters
    double m_N = 14.007;
    // double m_dummy = 0.0;
    // double M = 2*m_N;

    // Initial Energy
    double Ek = 0.0;
    // Calculate initial kinetic energy for convergence check
    for(int k=0; k<3; k+=2) { // Atom 0 and 2 have mass
         Ek += kinetic_energy(m_N, v[k*3], v[k*3+1], v[k*3+2]);
    }

    double Up_total = 0.0;
    // Force arrays
    double f[3][3]; // [atom][xyz]
    double fx_dummy=0, fy_dummy=0, fz_dummy=0, U_dummy=0;

    // Initial Force
    // Atom 0
    lennardjones(r[0], r[1], r[2], target, n_target, &f[0][0], &f[0][1], &f[0][2], &Up_total, lj_cutoff_sq);
    // Atom 2
    lennardjones(r[6], r[7], r[8], target, n_target, &f[2][0], &f[2][1], &f[2][2], &U_dummy, lj_cutoff_sq);
    Up_total += U_dummy;
    // Dummy (Atom 1) - No LJ
    f[1][0]=0; f[1][1]=0; f[1][2]=0;

    double Ei = Ek + Up_total;

    // COM Velocity for scattering angle
    double vcm_i[3];
    vcm_i[0] = 0.5*(v[0]+v[6]);
    vcm_i[1] = 0.5*(v[1]+v[7]);
    vcm_i[2] = 0.5*(v[2]+v[8]);

    double cur_dt = dt;
    int step = 0;

    while (step < max_steps) {

        // Prepare arrays for constraint solver
        double ri[3] = {r[0], r[1], r[2]};
        double rj[3] = {r[6], r[7], r[8]};
        double vi[3] = {v[0], v[1], v[2]};
        double vj[3] = {v[6], v[7], v[8]};

        // Distribute forces from dummy to N atoms (though dummy has 0 force here)
        // C++: fi = (1 - m1/2M)f0 + m0/M f1 - m1/2M f2 ...
        // Since m1=0, fi = f0, fj = f2.

        double fi[3] = {f[0][0], f[0][1], f[0][2]};
        double fj[3] = {f[2][0], f[2][1], f[2][2]};

        first_half_verlet_constrained(ri, rj, vi, vj, fi, fj, m_N, m_N, cur_dt, d_bond_sq);

        // Update positions/velocities in local array
        r[0]=ri[0]; r[1]=ri[1]; r[2]=ri[2];
        v[0]=vi[0]; v[1]=vi[1]; v[2]=vi[2];
        r[6]=rj[0]; r[7]=rj[1]; r[8]=rj[2];
        v[6]=vj[0]; v[7]=vj[1]; v[8]=vj[2];

        // Update Dummy (COM)
        r[3] = 0.5*(r[0]+r[6]);
        r[4] = 0.5*(r[1]+r[7]);
        r[5] = 0.5*(r[2]+r[8]);
        v[3] = 0.5*(v[0]+v[6]);
        v[4] = 0.5*(v[1]+v[7]);
        v[5] = 0.5*(v[2]+v[8]);

        // Calculate new forces
        Up_total = 0.0;
        lennardjones(r[0], r[1], r[2], target, n_target, &f[0][0], &f[0][1], &f[0][2], &Up_total, lj_cutoff_sq);
        lennardjones(r[6], r[7], r[8], target, n_target, &f[2][0], &f[2][1], &f[2][2], &U_dummy, lj_cutoff_sq);
        Up_total += U_dummy;

        // Update fi, fj
        fi[0]=f[0][0]; fi[1]=f[0][1]; fi[2]=f[0][2];
        fj[0]=f[2][0]; fj[1]=f[2][1]; fj[2]=f[2][2];

        second_half_verlet_constrained(ri, rj, vi, vj, fi, fj, m_N, m_N, cur_dt);

        r[0]=ri[0]; r[1]=ri[1]; r[2]=ri[2];
        v[0]=vi[0]; v[1]=vi[1]; v[2]=vi[2];
        r[6]=rj[0]; r[7]=rj[1]; r[8]=rj[2];
        v[6]=vj[0]; v[7]=vj[1]; v[8]=vj[2];

        // Dummy vel update
        v[3] = 0.5*(v[0]+v[6]);
        v[4] = 0.5*(v[1]+v[7]);
        v[5] = 0.5*(v[2]+v[8]);

        step++;

        // Check Bounds
        double rcm_x = r[3];
        double rcm_y = r[4];
        double rcm_z = r[5];

        // 1. Outside box
        if (fabs(rcm_x) >= lx || fabs(rcm_y) >= ly || fabs(rcm_z) >= lz) {
            // Check Energy Conservation
             Ek = 0.0;
             for(int k=0; k<3; k+=2) Ek += kinetic_energy(m_N, v[k*3], v[k*3+1], v[k*3+2]);
             double Ef = Ek + Up_total;
             if (fabs((Ef - Ei)/Ei) > 0.01) { // 1% tolerance
                 results[gid] = 2; // Lost
                 return;
             }

             // Calculate Chi
             double vcm_f[3] = {v[3], v[4], v[5]};
             double dot = vcm_i[0]*vcm_f[0] + vcm_i[1]*vcm_f[1] + vcm_i[2]*vcm_f[2];
             double mod_i = sqrt(vcm_i[0]*vcm_i[0] + vcm_i[1]*vcm_i[1] + vcm_i[2]*vcm_i[2]);
             double mod_f = sqrt(vcm_f[0]*vcm_f[0] + vcm_f[1]*vcm_f[1] + vcm_f[2]*vcm_f[2]);
             double arg = dot / (mod_i * mod_f);
             if (arg > 1.0) arg = 1.0;
             if (arg < -1.0) arg = -1.0;
             chi_results[gid] = acos(arg);
             results[gid] = 1; // Scatter
             return;
        }

        // 2. Outside Ellipsoid
        double ur = (rcm_x*rcm_x)/(a*a) + (rcm_y*rcm_y)/(b*b) + (rcm_z*rcm_z)/(c*c);
        if (ur > 1.0) {
             // Same check
             Ek = 0.0;
             for(int k=0; k<3; k+=2) Ek += kinetic_energy(m_N, v[k*3], v[k*3+1], v[k*3+2]);
             double Ef = Ek + Up_total;
             if (fabs((Ef - Ei)/Ei) > 0.01) {
                 results[gid] = 2; // Lost
                 return;
             }

             double vcm_f[3] = {v[3], v[4], v[5]};
             double dot = vcm_i[0]*vcm_f[0] + vcm_i[1]*vcm_f[1] + vcm_i[2]*vcm_f[2];
             double mod_i = sqrt(vcm_i[0]*vcm_i[0] + vcm_i[1]*vcm_i[1] + vcm_i[2]*vcm_i[2]);
             double mod_f = sqrt(vcm_f[0]*vcm_f[0] + vcm_f[1]*vcm_f[1] + vcm_f[2]*vcm_f[2]);
             double arg = dot / (mod_i * mod_f);
             if (arg > 1.0) arg = 1.0;
             if (arg < -1.0) arg = -1.0;
             chi_results[gid] = acos(arg);
             results[gid] = 1; // Scatter
             return;
        }
    }

    results[gid] = 2; // Lost (timeout)
}
