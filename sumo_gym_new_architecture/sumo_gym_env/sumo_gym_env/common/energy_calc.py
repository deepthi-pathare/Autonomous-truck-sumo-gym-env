import math

def calculate_energy_consumed(a, v, slope, duration):
    """
       Calculate the energy consumed by ego vehicle
    Args:
        a (float): Acceleration in m/s2
        v (float): Velocity in m/s
        slope (float): Slope of the current vehicle position in degrees
    """       
    m = 40000 # mass of ego vehicle in Kg
    Cd = 0.36 # Airdrag from tesla semi
    Af = 10 # m2
    Air_rho = 1.225 # kg/m3
    Cr = 0.005
    g = 9.81 # m/s2
    F = (m*a) + (0.5 * Cd * Af * Air_rho * v * v) + (g * Cr * m) + (m * g * math.sin(math.atan(slope/100)))
    P = F * v
    energy_consumed = (P/1000) * (duration/3600) # kwh
    return energy_consumed

def calculate_init_kinetic_energy(u):
    m = 40000 # mass of ego vehicle in Kg
    init_e = (0.5 * m * u * u) / (1000 * 3600)
    return init_e