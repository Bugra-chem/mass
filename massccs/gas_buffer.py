from .constants import ALPHA_TO_KCAL_MOL

class GasBuffer:
    def __init__(self, gas_flag=2):
        self.gas_flag = gas_flag
        self.natoms = 0
        self.gas_type = ""
        self.x = []
        self.y = []
        self.z = []
        self.q = []
        self.m = []
        self.eps = []
        self.sig = []
        self.mass = 0.0
        self.d = 0.0
        self.alpha_radial = 0.0
        self.alpha_axial = 0.0

        self.set_properties()

    def set_properties(self):
        if self.gas_flag == 2: # N2
            self.natoms = 3
            self.gas_type = "N2"

            # nitrogen 1
            self.x = [0.0, 0.0, 0.0]
            self.y = [0.0, 0.0, 0.0]
            self.z = [0.5488, 0.0, -0.5488]

            # Charges
            self.q = [-0.4825, 0.965, -0.4825]

            # Mass
            self.m = [14.007, 0.0, 14.007]

            # LJ
            self.eps = [1.0, 0.0, 1.0] # Dummy has 0
            self.sig = [0.0, 0.0, 0.0] # N2 LJ params handled via mixing rules in Force usually?
            # Wait, Force.cpp uses eps/sig from GasBuffer?
            # In C++ GasBuffer.cpp: eps[0]=1.0.
            # Force.cpp: lennardjones_LC uses moleculeTarget->eps/sig but for probe?
            # Actually C++ code for N2 uses `average_weighting_force_N2` or just `lennardjones_LC`?
            # `run_N2` calls `force->lennardjones(gasProbe, iatom, ...)`
            # `lennardjones` function takes `GasBuffer *gas`.
            # Inside `lennardjones`:
            # `epsilon = moleculeTarget->eps[i]; sigma = moleculeTarget->sig[i];`
            # It calculates LJ between probe atom and target atom.
            # But where are the probe parameters?
            # `lj1 = 4.0*epsilon*pow(sigma,6.0);`
            # It seems it uses ONLY target parameters? That's weird.
            # Ah, `MoleculeTarget::assignedParameter` returns specific parameters for the interaction with the gas?
            # Yes! "user_eps[i]*0.055" etc.
            # The eps/sig stored in MoleculeTarget are already the MIXED parameters (or effective parameters) for the specific gas type.
            # See MoleculeTarget.cpp: `if (gas_buffer_flag == 2) ...` sets up eps/sig for N2.
            # So GasBuffer's eps/sig are not used in `lennardjones`.

            self.mass = sum(self.m)
            self.d = abs(self.z[0] - self.z[2])

            self.alpha_radial = 2.19609742 * ALPHA_TO_KCAL_MOL
            self.alpha_axial = 1.51148405 * ALPHA_TO_KCAL_MOL
