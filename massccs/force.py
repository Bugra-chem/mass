import taichi as ti
from massccs.constants import *

@ti.func
def lennardjones_LC(
    r_probe: ti.types.vector(3, float),
    force_vec: ti.types.vector(3, float),
    lj_cutoff: float,

    # Linked Cell Data
    ti_atoms_inside_cell: ti.template(),
    ti_cell_atoms: ti.template(),
    ti_neighbors1_count: ti.template(),
    ti_neighbors1: ti.template(),

    # Target Data
    ti_target_x: ti.template(),
    ti_target_y: ti.template(),
    ti_target_z: ti.template(),
    ti_target_eps: ti.template(),
    ti_target_sig: ti.template(),

    # Grid params
    lx: float, ly: float, lz: float,
    corner: ti.types.vector(3, float),
    d_cell: float,
    Nx: int, Ny: int, Nz: int
):

    fx = 0.0
    fy = 0.0
    fz = 0.0
    U = 0.0

    # Calculate index
    xi = r_probe[0] - corner[0]
    i_idx = int(ti.floor(xi / d_cell))
    yi = r_probe[1] - corner[1]
    j_idx = int(ti.floor(yi / d_cell))
    zi = r_probe[2] - corner[2]
    k_idx = int(ti.floor(zi / d_cell))

    index = k_idx + Nz * j_idx + Nz * Ny * i_idx
    n_cells_total = Nx * Ny * Nz

    if index >= 0 and index < n_cells_total:
        n_neighbors = ti_neighbors1_count[index]

        for i in range(n_neighbors):
            cell_index = ti_neighbors1[index, i]
            atoms_in_cell = ti_atoms_inside_cell[cell_index]

            for j in range(atoms_in_cell):
                target_id = ti_cell_atoms[cell_index, j]

                dx = r_probe[0] - ti_target_x[target_id]
                dy = r_probe[1] - ti_target_y[target_id]
                dz = r_probe[2] - ti_target_z[target_id]
                r2 = dx*dx + dy*dy + dz*dz

                if r2 < lj_cutoff * lj_cutoff:
                    epsilon = ti_target_eps[target_id]
                    sigma = ti_target_sig[target_id]

                    r2inv = 1.0 / r2
                    r6inv = r2inv * r2inv * r2inv

                    sig2 = sigma * sigma
                    sig6 = sig2 * sig2 * sig2

                    lj1 = 4.0 * epsilon * sig6
                    lj2 = lj1 * sig6

                    Ulj = r6inv * (lj2 * r6inv - lj1)

                    rc2 = lj_cutoff * lj_cutoff
                    rc2inv = 1.0 / rc2
                    rc6inv = rc2inv * rc2inv * rc2inv
                    Ulj_cut = rc6inv * (lj2 * rc6inv - lj1)

                    lj3 = 6.0 * lj1
                    lj4 = 12.0 * lj2
                    flj = r6inv * (lj4 * r6inv - lj3) * r2inv

                    U += Ulj - Ulj_cut
                    fx += flj * dx
                    fy += flj * dy
                    fz += flj * dz

    return ti.Vector([fx, fy, fz]), U

@ti.func
def lennardjones_coulomb_LC(
    r_probe: ti.types.vector(3, float),
    q_probe: float,
    force_vec: ti.types.vector(3, float),
    lj_cutoff: float,
    coul_cutoff: float,

    # Linked Cell Data
    ti_atoms_inside_cell: ti.template(),
    ti_cell_atoms: ti.template(),
    ti_neighbors1_count: ti.template(),
    ti_neighbors1: ti.template(),
    ti_neighbors2_count: ti.template(),
    ti_neighbors2: ti.template(),

    # Target Data
    ti_target_x: ti.template(),
    ti_target_y: ti.template(),
    ti_target_z: ti.template(),
    ti_target_eps: ti.template(),
    ti_target_sig: ti.template(),
    ti_target_q: ti.template(),

    # Grid params
    lx: float, ly: float, lz: float,
    corner: ti.types.vector(3, float),
    d_cell: float,
    Nx: int, Ny: int, Nz: int,
    next_neighbor: int
):

    fx = 0.0
    fy = 0.0
    fz = 0.0
    U = 0.0

    xi = r_probe[0] - corner[0]
    i_idx = int(ti.floor(xi / d_cell))
    yi = r_probe[1] - corner[1]
    j_idx = int(ti.floor(yi / d_cell))
    zi = r_probe[2] - corner[2]
    k_idx = int(ti.floor(zi / d_cell))

    index = k_idx + Nz * j_idx + Nz * Ny * i_idx
    n_cells_total = Nx * Ny * Nz

    if index >= 0 and index < n_cells_total:
        n_neighbors = ti_neighbors1_count[index]
        for i in range(n_neighbors):
            cell_index = ti_neighbors1[index, i]
            atoms_in_cell = ti_atoms_inside_cell[cell_index]

            for j in range(atoms_in_cell):
                target_id = ti_cell_atoms[cell_index, j]

                dx = r_probe[0] - ti_target_x[target_id]
                dy = r_probe[1] - ti_target_y[target_id]
                dz = r_probe[2] - ti_target_z[target_id]
                r2 = dx*dx + dy*dy + dz*dz
                r = ti.sqrt(r2)

                if r < lj_cutoff:
                    epsilon = ti_target_eps[target_id]
                    sigma = ti_target_sig[target_id]
                    r2inv = 1.0 / r2
                    r6inv = r2inv * r2inv * r2inv
                    sig2 = sigma * sigma
                    sig6 = sig2 * sig2 * sig2
                    lj1 = 4.0 * epsilon * sig6
                    lj2 = lj1 * sig6
                    Ulj = r6inv * (lj2 * r6inv - lj1)
                    rc2 = lj_cutoff * lj_cutoff
                    rc2inv = 1.0 / rc2
                    rc6inv = rc2inv * rc2inv * rc2inv
                    Ulj_cut = rc6inv * (lj2 * rc6inv - lj1)
                    lj3 = 6.0 * lj1
                    lj4 = 12.0 * lj2
                    flj = r6inv * (lj4 * r6inv - lj3) * r2inv
                    U += Ulj - Ulj_cut
                    fx += flj * dx
                    fy += flj * dy
                    fz += flj * dz

                if r < coul_cutoff:
                    qj = ti_target_q[target_id]
                    rinv = 1.0 / r
                    Ucoul = q_probe * qj * rinv * KCOUL
                    Ucoul_shift = -1.5 * q_probe * qj * KCOUL / coul_cutoff + \
                                  0.5 * q_probe * qj * KCOUL * r2 / (coul_cutoff * coul_cutoff * coul_cutoff)
                    r2inv = 1.0 / r2
                    fcoul = Ucoul * r2inv * (1.0 - (r / coul_cutoff)**3)
                    U += Ucoul + Ucoul_shift
                    fx += fcoul * dx
                    fy += fcoul * dy
                    fz += fcoul * dz

        if next_neighbor == 1:
            n_neighbors2 = ti_neighbors2_count[index]
            for i in range(n_neighbors2):
                cell_index = ti_neighbors2[index, i]
                atoms_in_cell = ti_atoms_inside_cell[cell_index]

                for j in range(atoms_in_cell):
                    target_id = ti_cell_atoms[cell_index, j]
                    dx = r_probe[0] - ti_target_x[target_id]
                    dy = r_probe[1] - ti_target_y[target_id]
                    dz = r_probe[2] - ti_target_z[target_id]
                    r2 = dx*dx + dy*dy + dz*dz
                    r = ti.sqrt(r2)

                    if r < coul_cutoff:
                        qj = ti_target_q[target_id]
                        rinv = 1.0 / r
                        Ucoul = q_probe * qj * rinv * KCOUL
                        Ucoul_shift = -1.5 * q_probe * qj * KCOUL / coul_cutoff + \
                                      0.5 * q_probe * qj * KCOUL * r2 / (coul_cutoff * coul_cutoff * coul_cutoff)
                        r2inv = 1.0 / r2
                        fcoul = Ucoul * r2inv * (1.0 - (r / coul_cutoff)**3)
                        U += Ucoul + Ucoul_shift
                        fx += fcoul * dx
                        fy += fcoul * dy
                        fz += fcoul * dz

    return ti.Vector([fx, fy, fz]), U

@ti.func
def lennardjones_induced_dipole_LC(
    r_probe: ti.types.vector(3, float),
    force_vec: ti.types.vector(3, float),
    lj_cutoff: float,
    coul_cutoff: float,
    alpha: float,

    # Linked Cell Data
    ti_atoms_inside_cell: ti.template(),
    ti_cell_atoms: ti.template(),
    ti_neighbors1_count: ti.template(),
    ti_neighbors1: ti.template(),
    ti_neighbors2_count: ti.template(),
    ti_neighbors2: ti.template(),

    # Target Data
    ti_target_x: ti.template(),
    ti_target_y: ti.template(),
    ti_target_z: ti.template(),
    ti_target_eps: ti.template(),
    ti_target_sig: ti.template(),
    ti_target_q: ti.template(),

    # Grid params
    lx: float, ly: float, lz: float,
    corner: ti.types.vector(3, float),
    d_cell: float,
    Nx: int, Ny: int, Nz: int,
    next_neighbor: int
):
    fx = 0.0
    fy = 0.0
    fz = 0.0
    U = 0.0

    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    Exx = 0.0
    Eyy = 0.0
    Ezz = 0.0
    Exy = 0.0
    Exz = 0.0
    Eyz = 0.0

    xi = r_probe[0] - corner[0]
    i_idx = int(ti.floor(xi / d_cell))
    yi = r_probe[1] - corner[1]
    j_idx = int(ti.floor(yi / d_cell))
    zi = r_probe[2] - corner[2]
    k_idx = int(ti.floor(zi / d_cell))

    index = k_idx + Nz * j_idx + Nz * Ny * i_idx
    n_cells_total = Nx * Ny * Nz

    if index >= 0 and index < n_cells_total:
        n_neighbors = ti_neighbors1_count[index]
        for i in range(n_neighbors):
            cell_index = ti_neighbors1[index, i]
            atoms_in_cell = ti_atoms_inside_cell[cell_index]

            for j in range(atoms_in_cell):
                target_id = ti_cell_atoms[cell_index, j]

                dx = r_probe[0] - ti_target_x[target_id]
                dy = r_probe[1] - ti_target_y[target_id]
                dz = r_probe[2] - ti_target_z[target_id]
                r2 = dx*dx + dy*dy + dz*dz
                r = ti.sqrt(r2)
                r2inv = 1.0/r2

                if r < lj_cutoff:
                    epsilon = ti_target_eps[target_id]
                    sigma = ti_target_sig[target_id]
                    r6inv = r2inv * r2inv * r2inv
                    sig2 = sigma * sigma
                    sig6 = sig2 * sig2 * sig2
                    lj1 = 4.0 * epsilon * sig6
                    lj2 = lj1 * sig6
                    Ulj = r6inv * (lj2 * r6inv - lj1)

                    rc2 = lj_cutoff * lj_cutoff
                    rc2inv = 1.0 / rc2
                    rc6inv = rc2inv * rc2inv * rc2inv
                    Ulj_cut = rc6inv * (lj2 * rc6inv - lj1)

                    lj3 = 6.0 * lj1
                    lj4 = 12.0 * lj2
                    flj = r6inv * (lj4 * r6inv - lj3) * r2inv

                    U += Ulj - Ulj_cut
                    fx += flj * dx
                    fy += flj * dy
                    fz += flj * dz

                if r < coul_cutoff:
                    r3inv = (1.0 / r) * r2inv
                    r5inv = r3inv * r2inv
                    q = ti_target_q[target_id]
                    rc3inv = 1.0 / (coul_cutoff**3)
                    smooth_factor = (1.0 - r * r2 * rc3inv)
                    qr3inv = q * r3inv * smooth_factor
                    qr5inv = -3.0 * q * r5inv * smooth_factor
                    qrc = -3.0 * q * rc3inv * r2inv

                    Ex += dx * qr3inv
                    Ey += dy * qr3inv
                    Ez += dz * qr3inv

                    Exx += qr3inv + dx*dx*qr5inv + dx*dx*qrc
                    Eyy += qr3inv + dy*dy*qr5inv + dy*dy*qrc
                    Ezzi = qr3inv + dz*dz*qr5inv + dz*dz*qrc # Need to accumulate Ezz correctly, wait.
                    # Ezz accumulates. C++: Ezzi = ... Ezz += Ezzi.
                    Ezz += qr3inv + dz*dz*qr5inv + dz*dz*qrc

                    Exy += dx*dy*qr5inv + dx*dy*qrc
                    Exz += dx*dz*qr5inv + dx*dz*qrc
                    Eyz += dy*dz*qr5inv + dy*dz*qrc

        if next_neighbor == 1:
            n_neighbors2 = ti_neighbors2_count[index]
            for i in range(n_neighbors2):
                cell_index = ti_neighbors2[index, i]
                atoms_in_cell = ti_atoms_inside_cell[cell_index]

                for j in range(atoms_in_cell):
                    target_id = ti_cell_atoms[cell_index, j]

                    dx = r_probe[0] - ti_target_x[target_id]
                    dy = r_probe[1] - ti_target_y[target_id]
                    dz = r_probe[2] - ti_target_z[target_id]
                    r2 = dx*dx + dy*dy + dz*dz
                    r = ti.sqrt(r2)
                    r2inv = 1.0/r2

                    if r < coul_cutoff:
                        r3inv = (1.0 / r) * r2inv
                        r5inv = r3inv * r2inv
                        q = ti_target_q[target_id]
                        rc3inv = 1.0 / (coul_cutoff**3)
                        smooth_factor = (1.0 - r * r2 * rc3inv)
                        qr3inv = q * r3inv * smooth_factor
                        qr5inv = -3.0 * q * r5inv * smooth_factor
                        qrc = -3.0 * q * rc3inv * r2inv

                        Ex += dx * qr3inv
                        Ey += dy * qr3inv
                        Ez += dz * qr3inv

                        Exx += qr3inv + dx*dx*qr5inv + dx*dx*qrc
                        Eyy += qr3inv + dy*dy*qr5inv + dy*dy*qrc
                        Ezz += qr3inv + dz*dz*qr5inv + dz*dz*qrc

                        Exy += dx*dy*qr5inv + dx*dy*qrc
                        Exz += dx*dz*qr5inv + dx*dz*qrc
                        Eyz += dy*dz*qr5inv + dy*dz*qrc

    fx_ind = alpha * (Ex*Exx + Ey*Exy + Ez*Exz)
    fy_ind = alpha * (Ex*Exy + Ey*Eyy + Ez*Eyz)
    fz_ind = alpha * (Ex*Exz + Ey*Eyz + Ez*Ezz)

    fx += fx_ind
    fy += fy_ind
    fz += fz_ind

    U_ind = -0.5 * alpha * (Ex*Ex + Ey*Ey + Ez*Ez)
    U += U_ind

    return ti.Vector([fx, fy, fz]), U
