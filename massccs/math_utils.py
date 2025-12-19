import numpy as np
import math
import taichi as ti

# Taichi helper functions
@ti.func
def dot_product(v1: ti.types.vector(3, float), v2: ti.types.vector(3, float)) -> float:
    return v1.dot(v2)

@ti.func
def vec_modulus(v1: ti.types.vector(3, float)) -> float:
    return v1.norm()

@ti.func
def cross_product(v1: ti.types.vector(3, float), v2: ti.types.vector(3, float)) -> ti.types.vector(3, float):
    return v1.cross(v2)

@ti.func
def angle_vec(vi: ti.types.vector(3, float), vf: ti.types.vector(3, float)) -> float:
    t1 = vi.dot(vf)
    t2 = vi.norm()
    t3 = vf.norm()
    argument = t1 / (t2 * t3)

    # Clamp argument to [-1, 1] to avoid nan due to floating point errors
    if argument > 1.0: argument = 1.0
    if argument < -1.0: argument = -1.0

    return ti.acos(argument)

# Python helper functions (if needed on host)
def py_dot_product(v1, v2):
    return np.dot(v1, v2)

def py_vec_modulus(v1):
    return np.linalg.norm(v1)

def py_cross_product(v1, v2):
    return np.cross(v1, v2)
