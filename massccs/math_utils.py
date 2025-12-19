import math
import taichi as ti

@ti.func
def dot_product(v1: ti.template(), v2: ti.template()):
    return v1.dot(v2)

@ti.func
def vec_modulus(v1: ti.template()):
    return v1.norm()

@ti.func
def distance(v1: ti.template(), v2: ti.template()):
    return (v1 - v2).norm()

@ti.func
def cross_product(v1: ti.template(), v2: ti.template()):
    return v1.cross(v2)

@ti.func
def add_vec(v1: ti.template(), v2: ti.template()):
    return v1 + v2

@ti.func
def sub_vec(v1: ti.template(), v2: ti.template()):
    return v1 - v2

@ti.func
def normalize(v1: ti.template()):
    return v1.normalized()
