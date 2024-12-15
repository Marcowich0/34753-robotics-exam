import sympy as sp
import numpy as np
from sympy import pi, sin, cos
import matplotlib.pyplot as plt

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


def Rotation_matrix(angle, axis = 'z', deg = False):
    if deg:
        angle = sp.rad(angle)
    if axis == 'x':
        return sp.Matrix([[1, 0, 0, 0],
                          [0, cos(angle), -sin(angle), 0],
                          [0, sin(angle), cos(angle), 0],
                          [0, 0, 0, 1]])
    elif axis == 'y':
        return sp.Matrix([[cos(angle), 0, sin(angle), 0],
                          [0, 1, 0, 0],
                          [-sin(angle), 0, cos(angle), 0],
                          [0, 0, 0, 1]])
    elif axis == 'z':
        return sp.Matrix([[cos(angle), -sin(angle), 0, 0],
                          [sin(angle), cos(angle), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    else:
        raise ValueError('Invalid axis. Choose from x, y, z')
    

def find_denevit_hartenberg_perameters_from_transformation(T):
    theta_sol, d_sol, a_sol, alpha_sol = sp.symbols('theta d a alpha')
    Transformation = transformation_from_denavit_hartenberg([[theta_sol, d_sol, a_sol, alpha_sol]])
    sol = sp.solve(Transformation - T)[0]
    return sol

def Tranlation_matrix(trans):
    return sp.Matrix([[1,0,0,trans[0]],
                      [0,1,0,trans[1]],
                      [0,0,1,trans[2]],
                        [0,0,0,1]])


def Find_maxmin_in_function(f, var, domain):

    all_points = np.linspace(domain.start, domain.end, 100)
    # Evaluate f(t) at all points to find the maximum value
    v_values = [f.subs(var, point) for point in all_points]

    max = f.subs(var, all_points[np.argmax(v_values)])
    min = f.subs(var, all_points[np.argmin(v_values)])
    print(f"Maxima: { all_points[np.argmax(v_values)].evalf(3) } with value: {max} = {max.evalf(4)}")
    print(f"Minima: { all_points[np.argmin(v_values)].evalf(3) } with value: {min} = {min.evalf(4)}")

    return all_points[np.argmax(v_values)], np.max(v_values), all_points[np.argmin(v_values)], np.min(v_values)

def Transformation_matrix_from_3_points(P0, Px, Py):
    # P0: The origin, Px: A point on the x-axis, Py: A point in the xy-plane
    x = (Px - P0).normalized()
    y = (Py - P0).normalized()

    T = sp.eye(4)
    T[:3,0] = x
    T[:3,2] = x.cross(y).normalized()
    T[:3,1] = T[:3,2].cross(x)
    T[:3,3] = P0
    return T.evalf(4)


def Normalize_fraction(TF, size):
    num, den = sp.fraction(TF)
    return (num / size) / (den / size)