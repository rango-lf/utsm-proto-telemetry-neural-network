import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col
import pandas as pd
import math
import numpy as np
import scipy.integrate as integrate

factor_time = 0.2
factor_efficiency = 0.8

G = nx.DiGraph()
G.add_node("x0_k0",v=0) #add start node
coordinate_set_dict = {"x0_k0": (0,0)}

g = 9.81                    # [m/s^2] earth gravity constant


# Mechanical model of the car with the following constants:
driver_mass = 71            # [kg]   mass of the driver
car_mass = 123 + driver_mass# [kg]   weight of car and driver
diameter_tire = 0.4         # [m]    diameter of the tire
coeff_roll_res = 0.02       # [1]    Unitless roll resistance coefficient, the real value is possibly lower
gear_ratio = 5              # [1]    Unitless ratio of how many rotations the motor has to do for one roation of the wheel
coeff_aero_drag = 0.2       # [1]    Unitless aerodynamic Drag coefficient
A_aero = 0.8                # [m^2]  Aerodynamic front surface of the car
rho_air = 1.225             # [kg/m^3] Density of air at standard conditions 


# Battery electric car simulated by a model of series wound dc motor model with the follwing constants:
torque_max = 3.6            # [Nm] Engine maximum torque
torque_constant = 0.130     # [N*m/A]
velocity_constant = 70      # [1/(min * V)]
anchor_resistance = 0.1     # [Ohm]
torque_loss = 0.5           # [N*m]

section_length = 50         # [m]     distance resolution
velocity_spacing = 2        # [km/h]  velocity resolution of the simulation
max_acceleration = 4        # [1]     Unitless integer of how many velocity_spacing steps the car can accelerate or deaccelerte over one section_length

# Define the track profile using waypoints 
# Between waypoints missing the values are obtained with linear interpolation
waypoints = [
    {"s": 0,    "vmax": 0,  "vmin": 0,  "m_above_sea": 100}, # waypoint at start of track needed
    {"s": 50,   "vmax": 30, "vmin": 5, "m_above_sea": 101},
    {"s": 100,  "vmax": 50, "vmin": 10, "m_above_sea": 101.3},
    {"s": 500,  "vmax": 50, "vmin": 20, "m_above_sea": 101},
    {"s": 600,  "vmax": 50, "vmin": 20, "m_above_sea": 102},
    {"s": 750,  "vmax": 50, "vmin": 20, "m_above_sea": 103},
    {"s": 850,  "vmax": 50, "vmin": 20, "m_above_sea": 100},
    {"s": 1599, "vmax": 50, "vmin": 0,  "m_above_sea": 100}, # allow speed to drop again before the end
    {"s": 1600, "vmax": 0,  "vmin": 0 , "m_above_sea": 100}  # waypoint at end of track needed
]

# Create the full track profile out of the waypoints above
waypoint_df = pd.DataFrame(waypoints).set_index("s").sort_index()
s_range = np.arange(waypoint_df.index.min(), waypoint_df.index.max() + 1, section_length)
config = waypoint_df.reindex(waypoint_df.index.union(s_range)).sort_index().interpolate("index").loc[s_range]
config = config.reset_index()


x_max = len(config.index)-1  # index of the end of the last section

# Helper fuction to calculate the Gradient angle inside section x-1
def get_gradient_angle(x):
    return math.atan( (config.loc[x, "m_above_sea"]-config.loc[x-1, "m_above_sea"] ) / section_length)   #in rad

print(config)

class ImpossibleState(Exception):
    pass


class PhysicalCarModel:
    def __init__(self):
        pass                                          # No initalisation needed

    
    def predict(self,gradient_angle, v1, v2, section_length):   

        
        accel = (v2**2 - v1**2) / section_length / 2. # [m/s^2] acceleration constant
        
        if math.isclose(accel, 0.):
            time = section_length/v2                  # When not acclerated speed stay constant
        else:
            time = (v2 - v1) / accel                  # time for the section
    
        def velocity(t):
            return v1 + accel*t
        
        def total_force(t):
            # Rolling resistance while going up (or down) a slope
            f_roll_res = car_mass * g * math.cos(gradient_angle) * coeff_roll_res
            
            # Force needed for going up (or down) a slope
            f_slope = car_mass * g * math.sin(gradient_angle)
            
            # Forces needed to accelerate the car
            f_accel = accel * car_mass
            
            # Aerodynamic forces depend on time
            f_aero_drag = 0.5 * rho_air * A_aero * coeff_aero_drag * velocity(t)**2
            return f_roll_res + f_slope + f_accel + f_aero_drag
        
        def total_power_usage(t):
            return total_force(t)*velocity(t)         
    
        def engine_rotation_speed(t):   # [rad/s] Motor rotation speed
            return velocity(t) * gear_ratio / (diameter_tire/2) 
        
        def engine_torque(t):           # [Nm]    Motor torque
            torque = total_power_usage(t)/engine_rotation_speed(t)
            if torque > torque_max:     # Don't allow torque to be too high
                raise ImpossibleState("Torque exceeds the maximum allowed.")
            return torque
    
        def engine_current(t):          # [A]     Current
            return (engine_torque(t)+torque_loss) / torque_constant
    
        def engine_voltage(t):          # [V]     Voltage
            factor_rotations_per_minute = 60 / (2 * math.pi)
            return (engine_rotation_speed(t) * factor_rotations_per_minute / velocity_constant) + (anchor_resistance * engine_torque(t))
    
        def engine_power(t):
            engine_power = engine_voltage(t) * engine_current(t)
            if total_power_usage(t) < 0: # Shut off engine if no power needed
                engine_power = 0
            return engine_power
            
        Energy_Motor, _ = integrate.quad(engine_power, 0, time)
        return Energy_Motor, time


# Use physical formulas to simulate the behaviour of the car
model = PhysicalCarModel()

for x in range (1, x_max+1):   
    # change vmin and vmax from the track config to discrete boundaries
    k_min = int(np.ceil(config.loc[x, "vmin"] / velocity_spacing))
    k_max = int(np.floor(config.loc[x, "vmax"] / velocity_spacing))

    for k in range(k_min, k_max + 1):
        v = k * velocity_spacing
        node_name = f"x{x}_k{k}"
        G.add_node((node_name), v=v)
        coordinate_set_dict[node_name] = (x*section_length, v) # add to dictionary to later print it at correct position
        
        for kold in range(k-max_acceleration,k+max_acceleration+1):
            if G.has_node(f"x{x-1}_k{kold}"):
                vold = kold*velocity_spacing

                gradient_angle = get_gradient_angle(x)
                vold_mps       = vold / 3.6
                v_mps          = v / 3.6            

                try:
                    # Use the car model to predict the Energy and time needed to pass the section
                    E_Engine, time = model.predict(gradient_angle, vold_mps, v_mps, section_length)
                except ImpossibleState:
                    # Transition not possible
                    continue  # Skip this iteration

                # As adding up speed and fuel usage is comparing apples and oranges, we have to prescale what we calculated 
                # so that the speed and efficiency factors have the effect we hope for.
                cost_econ = E_Engine / 2500
                cost_speed = time / 10

                edge_weight = factor_time*cost_speed + factor_efficiency*cost_econ
                G.add_edge(f"x{x-1}_k{kold}", node_name, weight=edge_weight, time=time, energy=E_Engine)

x_end = len(config.index)-1
k_end = int(np.ceil(config.loc[len(config.index)-1]["vmin"] / velocity_spacing))

nodelist = nx.dijkstra_path(G, "x0_k0",f"x{x_end}_k{k_end}")

outputlist = []
for i in range(len(config.index)):
     outputlist.append(G.nodes[nodelist[i]]["v"])

output = pd.DataFrame(outputlist)
output.rename(columns={0:"Speed optimal"}, inplace = True)
output['distance'] = output.index * section_length
print("Best Profile:", output)


total_time = 0
for u, v in zip(nodelist[:-1], nodelist[1:]):
    total_time += G.edges[u, v].get("time", 0)
print("Total time (seconds):", total_time)


total_energy = 0
for u, v in zip(nodelist[:-1], nodelist[1:]):
    total_energy += G.edges[u, v].get("energy", 0)
print("Total energy (joules):", total_energy)

"""
matrix = nx.to_pandas_adjacency(G)
matrix.to_csv("AdjMatrix.csv", sep=',')
"""


weight_list = []
attributelist = nx.get_edge_attributes(G,'weight')
for edge in G.edges:
    weight_list.append(attributelist[edge])


fig, ax = plt.subplots(figsize=(10, 6))

nx.draw(
    G,
    pos=coordinate_set_dict,
    ax=ax,
    with_labels=False,
    node_color='blue',
    edge_color='lightgrey',
    node_size=2,
)
ax.plot(output['distance'], output['Speed optimal'], color='blue', linewidth=2, label='Optimal Speed')
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Speed [km/h]')
ax.set_title('Optimal Path')
ax.axis('on')
plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True, labelleft=True)
plt.show()
