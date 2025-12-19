import json
import sys
import multiprocessing
from .constants import *

class InputParser:
    def __init__(self, filename):
        self.filename = filename
        self.data = self._parse_file(filename)
        self._validate_and_set_params()

    def _parse_file(self, filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {filename}.")
            sys.exit(1)

    def _validate_and_set_params(self):
        d = self.data

        # Required Parameters
        if "targetFileName" not in d:
            print("Error input.json: targetFileName not found")
            sys.exit(1)
        self.target_filename = d["targetFileName"]

        # Optional Parameters
        self.n_probe = d.get("numberProbe", NPROBE)
        self.n_iter = d.get("nIter", NITER)
        self.seed = d.get("seed", SEED)
        self.n_threads = d.get("nthreads", multiprocessing.cpu_count())
        self.dt = d.get("dt", TIMESTEP)
        self.temperature_target = d.get("Temp", TEMPERATURE)
        self.skin = d.get("skin", SKIN)

        # Gas Buffer
        self.gas_buffer_str = d.get("GasBuffer", "He")
        if self.gas_buffer_str == "He":
            self.gas_buffer_flag = 1
        elif self.gas_buffer_str == "N2":
            self.gas_buffer_flag = 2
        elif self.gas_buffer_str == "CO2":
            self.gas_buffer_flag = 3
        elif self.gas_buffer_str == "Ar":
            self.gas_buffer_flag = 4
        elif self.gas_buffer_str == "co2":
            self.gas_buffer_flag = 5
        else:
            print("only available the follow gas buffer types: He Ar N2 CO2")
            sys.exit(1)

        # Equipotential
        self.equipotential_str = d.get("Equipotential", "no")
        self.equipotential_flag = 1 if self.equipotential_str == "yes" else 0

        # Short-range cutoff
        self.short_range_str = d.get("Short-range cutoff", "yes")
        self.short_range_cutoff = 1 if self.short_range_str == "yes" else 0
        self.lj_cutoff = d.get("LJ-cutoff", SHORT_CUTOFF)

        # Long-range forces
        self.long_range = d.get("Long-range forces", "no")
        self.long_range_flag = 1 if self.long_range == "yes" else 0

        # Long-range cutoff
        self.long_range_str = d.get("Long-range cutoff", "yes")
        self.long_range_cutoff = 1 if self.long_range_str == "yes" else 0
        self.coul_cutoff = d.get("Coul-cutoff", LONG_CUTOFF)

        # Polarizability
        if "polarizability" in d:
            self.polarizability_str = d["polarizability"]
            self.polarizability_flag = 1 if self.polarizability_str == "yes" else 0
        else:
            if self.gas_buffer_flag == 1:
                self.polarizability_str = "yes"
                self.polarizability_flag = 1
            else:
                self.polarizability_str = "no"
                self.polarizability_flag = 0

        # Alpha
        if "alpha" in d:
            self.alpha = d["alpha"]
        else:
            if self.polarizability_flag == 1:
                if self.gas_buffer_flag == 1: self.alpha = ALPHA_HE
                elif self.gas_buffer_flag == 2: self.alpha = ALPHA_N2
                elif self.gas_buffer_flag == 3: self.alpha = ALPHA_CO2
                elif self.gas_buffer_flag == 4: self.alpha = ALPHA_AR
                elif self.gas_buffer_flag == 5: self.alpha = ALPHA_co2
                else: self.alpha = 0.0
            else:
                self.alpha = 0.0

        # Force field
        if "force-field" in d:
            self.user_ff = d["force-field"]
            self.user_ff_flag = 1
        else:
            self.user_ff = ""
            self.user_ff_flag = 0

    def print_read_input(self):
        print("*********************************************************")
        print("INPUT:: Simulation Parameters ")
        print("*********************************************************")
        print(f"target filename                  : {self.target_filename}")
        print(f"number of probe                  : {self.n_probe}")
        print(f"number of iterarions             : {self.n_iter}")
        print(f"number of threads                : {self.n_threads}")
        print(f"seed number                      : {self.seed}")
        print(f"gas buffer                       : {self.gas_buffer_str}")
        print(f"Target Temperature (K)           : {self.temperature_target}")
        print(f"timestep (fs)                    : {self.dt}")
        print(f"Skin cell size (Ang)             : {self.skin}")
        print(f"Equipotential                    : {self.equipotential_str}")
        print(f"Cut short-range interaction      : {self.short_range_str}")
        if self.short_range_cutoff == 1:
            print(f"LJ cutoff (Ang)                  : {self.lj_cutoff}")
        print(f"Apply long-range interaction     : {self.long_range}")
        print(f"Cut long-range interaction       : {self.long_range_str}")
        if self.long_range_cutoff == 1:
            print(f"Coulomb cutoff (Ang)             : {self.coul_cutoff}")
        print(f"Apply induced-dipole interaction : {self.polarizability_str}")
        if self.polarizability_flag == 1:
            print(f"alpha (Ang^3)                    : {self.alpha}")
        if self.user_ff_flag == 1:
            print(f"force-field                      : {self.user_ff}")
