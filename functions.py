import sympy as sp
import numpy as np
from sympy import pi, sin, cos

def skew_matrix(v):
    return sp.Matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
def unskew_matrix(v):
    return sp.Matrix([v[2, 1], v[0, 2], v[1, 0]])

def transformation_from_denavit_hartenberg(DH_list, deg = False):
    A = sp.eye(4)
    for peram in DH_list:
        theta, d, a, alpha = peram
        if deg:
            theta = sp.rad(theta)
            alpha = sp.rad(alpha)
        Trans = sp.Matrix([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                           [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                           [0, sin(alpha), cos(alpha), d],
                           [0, 0, 0, 1]])
        
        A = A * sp.simplify(Trans)
    return sp.simplify(A)

def Jacobian_from_denavit_hartenberg(DH_list, deg = False):
    N = len(DH_list)
    T_arr = []
    for i in range(N+1):
        T_arr.append(transformation_from_denavit_hartenberg(DH_list[:i], deg))
    
    Jac = sp.zeros(6, N)
    o_n = T_arr[-1][:3, 3]
    for i, (T, peram) in enumerate(zip(T_arr[:-1], DH_list)):
        revolute = isinstance(peram[0], sp.Basic) or isinstance(peram[3], sp.Basic) # if joint is revolute
        prismatic = isinstance(peram[1], sp.Basic) or isinstance(peram[2], sp.Basic) # if joint is prismatic
        if (revolute): 
            z = T[:3, 2]
            o = T[:3, 3]
            Jac[:, i] = sp.Matrix([z.cross(o_n - o), z])
        elif (prismatic): 
            z = T[:3, 2]
            Jac[:, i] = sp.Matrix([z, sp.zeros(3, 1)])
    return Jac



        

        