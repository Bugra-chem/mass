import json
from .constants import ALPHA_HE

class InputParameters:
    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            data = json.load(f)

        self.target_filename = data.get("targetFileName")
        self.n_probe = data.get("numberProbe", 10000)
        self.n_iter = data.get("nIter", 10)
        self.seed = data.get("seed", 2104)
        self.dt = data.get("dt", 10.0)
        self.temperature_target = data.get("Temp", 298.0)
        self.skin = data.get("skin", 0.01)
        self.gas_buffer_type = data.get("GasBuffer", "He")
        self.lj_cutoff = data.get("LJ-cutoff", 12.0)
        self.coul_cutoff = data.get("Coul-cutoff", 25.0)
        self.alpha = data.get("alpha", ALPHA_HE)
        self.polarizability = data.get("polarizability", "no") == "yes"
        self.short_range_cutoff = data.get("Short-range cutoff", "yes") == "yes"
        self.long_range = data.get("Long-range forces", "no") == "yes"
        self.long_range_cutoff = data.get("Long-range cutoff", "yes") == "yes"

        # Force type logic
        self.force_type = 2
        if self.gas_buffer_type in ["He", "Ar", "co2"]:
             if not self.short_range_cutoff and not self.long_range:
                 self.force_type = 1
             elif self.short_range_cutoff and not self.long_range:
                 self.force_type = 2
             elif not self.short_range_cutoff and self.polarizability and not self.long_range_cutoff:
                 self.force_type = 3
             elif self.short_range_cutoff and self.polarizability and self.long_range_cutoff:
                 self.force_type = 4

    def print_summary(self):
        print("*********************************************************")
        print("INPUT:: Simulation Parameters ")
        print("*********************************************************")
        print(f"target filename                  : {self.target_filename}")
        print(f"number of probe                  : {self.n_probe}")
        print(f"number of iterarions             : {self.n_iter}")
        print(f"seed number                      : {self.seed}")
        print(f"gas buffer                       : {self.gas_buffer_type}")
        print(f"Target Temperature (K)           : {self.temperature_target}")
        print(f"timestep (fs)                    : {self.dt}")
        print(f"Skin cell size (Ang)             : {self.skin}")
        print(f"LJ cutoff (Ang)                  : {self.lj_cutoff}")
        print(f"Coulomb cutoff (Ang)             : {self.coul_cutoff}")
        print(f"alpha (Ang^3)                    : {self.alpha}")
        print(f"Force Type                       : {self.force_type}")
