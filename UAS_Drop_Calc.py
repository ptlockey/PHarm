#Latest version including a reduction in force as UA accelerates due to wind
#To make wind act in the opposite direction, make F a negative sign, and add vx in the Force calculation 

import numpy as np
import math

# Input values from the user
initial_height = float(input("Enter initial height (meters): "))
m = float(input("Enter mass (kg): "))
u = float(input("Enter wind speed (m/s): "))
surface_pressure_mb = float(input("Enter surface pressure (mb): "))
surface_temp_celsius = float(input("Enter surface temperature (°C): "))
#relative_humidity = float(input("Enter relative humidity at the surface (%): "))
#Cd = float(input("Enter drag coefficient: "))
#A = float(input("Enter cross-sectional area (m^2): "))
v0 = float(input("Enter initial velocity (m/s): "))
CdA = float(input("Enter the CdA value : "))

# Convert surface temperature from Celsius to Kelvin
surface_temp_kelvin = surface_temp_celsius + 273.15

# Define the time step (adjust as needed)
dt = 0.001  # Time step for numerical integration (s)

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
R = 287.05  # Universal gas constant (J/(kg*K))
L = 0.0065  # Temperature lapse rate (K/m)
#CdA = 0.116 # Value for the M4T CdA as derived from experiment

# Calculate initial pressure at the surface
surface_pressure = surface_pressure_mb * 100  # Convert mb to Pascals

# Initialize initial temperature and density values
#temperature_values = [surface_temp_kelvin]

# Initialize air_density with the value at the specified initial height
current_temp_kelvin = surface_temp_kelvin - L * initial_height
current_pressure = surface_pressure * (1 - (L * initial_height) / surface_temp_kelvin) ** (g / (R * L))
initial_air_density = current_pressure / (R * current_temp_kelvin)

# Create lists to store air density and temperature values
air_density_values = [initial_air_density]
temperature_values = [current_temp_kelvin]

# Initial conditions
x = 0.0
z = initial_height
vx = 0.0  # Initial horizontal velocity due to wind in the x-axis set to zero
vz = 0.0  #Initial vertical velocity in the z-axis
x_wind = 0.0  # Initialize x-position due to wind force alone
vxx = v0  #Initial horizontal velocity due to UA motion only

# Initialize time
time = 0.0

# Lists to store trajectory data
x_values = []
z_values = []
x_wind_values = []  # To track x-position due to wind force only

# Numerical integration using Euler's method
while z >= 0:
    # Calculate temperature at the current altitude (y)
    current_temp_kelvin = surface_temp_kelvin - L * z
    temperature_values.append(current_temp_kelvin)  # Store temperature at this altitude

    # Calculate pressure at the current altitude (y)
    current_pressure = surface_pressure * (1 - (L * z) / surface_temp_kelvin) ** (g / (R * L))

    # Calculate air density at the current altitude (y) using the ideal gas law
    air_density = current_pressure / (R * current_temp_kelvin)
    air_density_values.append(air_density)  # Store air density at this altitude

    # Calculate terminal velocity at this altitude
    u_term = math.sqrt((2 * m * g) / (air_density * CdA))

    # Calculate drag force F and k using air density at altitude y
    F = (1 / 2) * air_density * (u-vx) ** 2 * CdA
    k = (air_density * CdA) / (2 * m)

    # Calculate acceleration in x and y directions
    v = np.sqrt(vx**2 + vz**2)  #v is used to calculate a velocity vector 
    v2 = np.sqrt(vxx**2 + vz**2)  #v2 is a seperate velocity calculation in the x-axis for the UA velocity at failure
    ax = (F / m) #Acceleration in the x-axis due to wind force
    az = -g - (k * v * vz)  #Acceleration in the z-axis due to gravity, taking air resistance into account. Note velocity is a vector
    bx = -(k * v2 * vxx)  #a seperate deceleration calculation to reduce UA initial velcoity due to air resistance. Note velocity is a vector

    # Update velocities in x and y directions
    vx += ax * dt #Velocity due to wind
    vz += az * dt #Velocity in the z axis due to gravity
    vxx += bx * dt #Deceleration due to drag forces

    # Ensure vx does not exceed wind speed (u)
    vx = min(vx, u)

    # Limit vy to terminal velocity (u_term)
    if vz > u_term:
        vz = u_term

    # Update positions in x and y directions
    x += vx * dt + vxx * dt  #Velocity in the x-axis is the sum of wind / initial velocities
    z += vz * dt

    # Update x-position due to wind force alone
    x_wind += (vx * dt)

    # Update time
    time += dt

    # Store values for plotting
    x_values.append(x)
    z_values.append(z)
    x_wind_values.append(x_wind)

# Calculate the final x-position, final velocity, and x-position due to wind force
final_x = x    #Variable to store the final x position
final_vx = vx  # Final x-velocity when y=0
final_vz = vz  # Final y-velocity when y=0
final_v = np.sqrt(final_vx ** 2 + final_vz ** 2)  # Total velocity
xvx = x - x_wind  # Final x-position due to initial velocity
final_vxvz = np.sqrt(final_vx**2 + final_vz**2)
GRZ = x_wind**2 * 3.141592653589793 #Create a variable to store the GRZ area, which is a radius due to wind centerd on xvx
GRZ_Min = xvx-x_wind #Variable to store the GRZ minimum distance
Pcoll = 1/GRZ #Probability of a single person being hit within the GRZ

# Print the results
print(f"Initial Temperature at {initial_height} meters: {temperature_values[0]:.2f} K")
print(f"Initial Air Density at {initial_height} meters: {air_density_values[0]:.4f} kg/m^3")
print(f"Final x-position: {final_x:.2f} meters")
print(f"X-position due to initial velocity only: {xvx:.2f} m")
print(f"Final x-position due to wind force alone: {x_wind:.2f} meters")
print(f"Final velocity (x): {final_vx:.2f} m/s")
print(f"Final velocity (z): {final_vz:.2f} m/s")
print(f"Final velocity (x,z): {final_vxvz:.2f} m/s")
print(f"Total time until z=0: {time:.2f} seconds")
print(f"GRZ Origin Position = {xvx:.2f} metres")
print(f"GRZ Area around the origin position = {GRZ:.2f} metres squared")
print(f"GRZ radius around the origin point = {x_wind:.2f} metres")
print(f"GRZ maximum distance = {final_x:.2f} metres")
print(f"GRZ minimum distance = {GRZ_Min:.2f} metres")
print(f"Probability of collision with 1 person = {Pcoll:.5f}")
