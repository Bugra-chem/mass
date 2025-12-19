import json
import sys
import multiprocessing
from massccs.constants import *

class Input:
    def __init__(self, input_file):
        self.input_file = input_file
        self.d = {}

        # Default values
        self.nProbe = NPROBE
        self.nIter = NITER
        self.seed = SEED
        self.nthreads = multiprocessing.cpu_count()
        self.targetFilename = ""
        self.user_ff = ""
        self.user_ff_flag = 0
        self.dt = TIMESTEP
        self.temperatureTarget = TEMPERATURE
        self.skin = SKIN
        self.gas_buffer_flag = 1
        self.polarizability_flag = 0
        self.equipotential_flag = 0
        self.short_range_cutoff = 1
        self.lj_cutoff = SHORT_CUTOFF
        self.long_range_flag = 0
        self.long_range_cutoff = 1
        self.coul_cutoff = LONG_CUTOFF
        self.alpha = 0.0

        # String representations for printing
        self.gas_buffer_str = "He"
        self.equipotential_str = "no"
        self.short_range_str = "yes"
        self.long_range = "no"
        self.long_range_str = "yes"
        self.polarizability_str = "no"

        self.readInputFile(input_file)

    def readInputFile(self, input_file):
        try:
            with open(input_file, 'r') as f:
                self.d = json.load(f)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)

        if "targetFileName" in self.d:
            self.targetFilename = self.d["targetFileName"]
        else:
            print("Error input.json: targetFileName not found")
            sys.exit(1)

        if "numberProbe" in self.d:
            self.nProbe = self.d["numberProbe"]

        if "nIter" in self.d:
            self.nIter = self.d["nIter"]

        if "seed" in self.d:
            self.seed = self.d["seed"]

        if "nthreads" in self.d:
            self.nthreads = self.d["nthreads"]

        if "dt" in self.d:
            self.dt = self.d["dt"]

        if "Temp" in self.d:
            self.temperatureTarget = self.d["Temp"]

        if "skin" in self.d:
            self.skin = self.d["skin"]

        if "GasBuffer" in self.d:
            self.gas_buffer_str = self.d["GasBuffer"]
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

        if "Equipotential" in self.d:
            self.equipotential_str = self.d["Equipotential"]
            if self.equipotential_str == "yes":
                self.equipotential_flag = 1
            elif self.equipotential_str == "no":
                self.equipotential_flag = 0
            else:
                print("need to choice Equipotential: yes or no")
                sys.exit(1)

        if "Short-range cutoff" in self.d:
            self.short_range_str = self.d["Short-range cutoff"]
            if self.short_range_str == "yes":
                self.short_range_cutoff = 1
            elif self.short_range_str == "no":
                self.short_range_cutoff = 0
            else:
                print("need to choice Short-range cutoff: yes or no")
                sys.exit(1)

        if "LJ-cutoff" in self.d:
            self.lj_cutoff = self.d["LJ-cutoff"]

        if "Long-range forces" in self.d:
            self.long_range = self.d["Long-range forces"]
            if self.long_range == "yes":
                self.long_range_flag = 1
            elif self.long_range == "no":
                self.long_range_flag = 0
            else:
                print("need specify only yes or no for coulomb interactions")
                sys.exit(1)

        if "Long-range cutoff" in self.d:
            self.long_range_str = self.d["Long-range cutoff"]
            if self.long_range_str == "yes":
                self.long_range_cutoff = 1
            elif self.long_range_str == "no":
                self.long_range_cutoff = 0
            else:
                print("need to choice Long-range cutoff: yes or no")
                sys.exit(1)

        if "Coul-cutoff" in self.d:
            self.coul_cutoff = self.d["Coul-cutoff"]

        if "polarizability" in self.d:
            self.polarizability_str = self.d["polarizability"]
            if self.polarizability_str == "yes":
                self.polarizability_flag = 1
            elif self.polarizability_str == "no":
                self.polarizability_flag = 0
            else:
                print("need to choice polarizability: yes or no")
                sys.exit(1)
        else:
            if self.gas_buffer_flag == 1:
                self.polarizability_str = "yes"
                self.polarizability_flag = 1
            else:
                self.polarizability_str = "no"
                self.polarizability_flag = 0

        if "alpha" in self.d:
            self.alpha = self.d["alpha"]
        else:
            if self.polarizability_flag == 1:
                if self.gas_buffer_flag == 1: self.alpha = ALPHA_HE
                elif self.gas_buffer_flag == 2: self.alpha = ALPHA_N2
                elif self.gas_buffer_flag == 3: self.alpha = ALPHA_CO2
                elif self.gas_buffer_flag == 4: self.alpha = ALPHA_AR
                elif self.gas_buffer_flag == 5: self.alpha = ALPHA_co2
                else: self.alpha = 0.0

        if "force-field" in self.d:
            self.user_ff = self.d["force-field"]
            self.user_ff_flag = 1
        else:
            self.user_ff_flag = 0

    def printReadInput(self):
        print("*********************************************************")
        print("INPUT:: Simulation Parameters ")
        print("*********************************************************")
        print(f"target filename                  : {self.targetFilename}")
        print(f"number of probe                  : {self.nProbe}")
        print(f"number of iterarions             : {self.nIter}")
        print(f"number of threads                : {self.nthreads}")
        print(f"seed number                      : {self.seed}")
        print(f"gas buffer                       : {self.gas_buffer_str}")
        print(f"Target Temperature (K)           : {self.temperatureTarget}")
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
