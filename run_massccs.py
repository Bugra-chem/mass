import sys
import numpy as np
from massccs.input_parser import InputParameters
from massccs.molecule import Molecule
from massccs.core import calculate_reduced_mass, calculate_ellipsoid
from massccs.cpu_engine import CPUEngine
from massccs.gpu_engine import GPUEngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_massccs.py input.json [--cpu]")
        sys.exit(1)

    input_file = sys.argv[1]
    force_cpu = "--cpu" in sys.argv

    print(f"Reading input from {input_file}")
    params = InputParameters(input_file)
    params.print_summary()

    print(f"Reading molecule from {params.target_filename}")
    molecule = Molecule(params.target_filename, params.gas_buffer_type)

    reduced_mass = calculate_reduced_mass(molecule.mass_sum, params.gas_buffer_type)
    print(f"Reduced Mass: {reduced_mass} amu")

    ellipsoid_params = calculate_ellipsoid(molecule, params.lj_cutoff, params.skin)
    print(f"Ellipsoid axes: {ellipsoid_params[:3]}")
    print(f"Max impact parameter bmax: {ellipsoid_params[3]}")

    if force_cpu:
        print("Using CPU Engine")
        engine = CPUEngine(molecule, params, reduced_mass, ellipsoid_params)
    else:
        try:
            print("Attempting to use GPU Engine...")
            engine = GPUEngine(molecule, params, reduced_mass, ellipsoid_params)
        except Exception as e:
            print(f"GPU Init failed: {e}")
            print("Falling back to CPU Engine")
            engine = CPUEngine(molecule, params, reduced_mass, ellipsoid_params)

    # Run
    chi, success = engine.run()

    # Analyze
    bmax = ellipsoid_params[3]
    n_total = len(chi)

    # Filter valid
    # success: 1=scattered, 0=missed/free, -1=lost

    # Calculate Omega
    # Omega = (pi * bmax^2 / N_total) * Sum(1 - cos(chi))
    # We must include ALL trajectories in N_total to represent the integral over probability space
    # (assuming P(b) ~ b distribution we generated matches standard MC integration)

    # Generation: b = sqrt(bmax^2 * rnd). This generates b distributed as P(b) ~ 2b/bmax^2.
    # This is the correct importance sampling for area integration 2*pi*b db.
    # Integral = Area * Average(Value).
    # Area = pi * bmax^2.
    # Value = 1 - cos(chi).
    # Average is over ALL sampled points.
    # If missed (success=0), chi is undefined/irrelevant but actually cos(chi)=1 -> value=0.
    # If lost (success=-1), typically ignored or counted as error.

    valid_indices = np.where(success >= 0)[0] # Include success and missed
    chi_valid = chi[valid_indices]
    success_valid = success[valid_indices]

    # For missed trajectories (success=0), scattering angle is effectively 0 -> 1-cos(0) = 0.
    # So we just sum (1-cos(chi)) for success=1.

    sum_term = np.sum(1.0 - np.cos(chi[success == 1]))

    # If we ignore lost trajectories from N_total:
    N_effective = len(valid_indices)

    if N_effective == 0:
        print("No valid trajectories!")
        return

    ccs = (np.pi * bmax**2 / N_effective) * sum_term

    print("*********************************************************")
    print(f"Total Trajectories: {n_total}")
    print(f"Scattered: {np.sum(success == 1)}")
    print(f"Missed (Free): {np.sum(success == 0)}")
    print(f"Lost: {np.sum(success == -1)}")
    print(f"Calculated CCS: {ccs} Ang^2")
    print("*********************************************************")

if __name__ == "__main__":
    main()
