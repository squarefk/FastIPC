import math

def suggestedDt(E, nu, rho, dx, cfl):
    elasticity = math.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))
    return cfl * dx / elasticity