# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 00:15:55 2022

@author: Admin
"""

import sys
import os 
import pandas as pd 
import pypsa
from pypsa.linopf import network_lopf
import matplotlib.pyplot as plt
import networkx as nx
import numpy  as np
import netCDF4 as nc
import itertools

# Reading data files 
os.chdir(r"G:\My Drive\Master thesis\Data")
# =============================================================================
# os.chdir(r"/home/yalmousa1/Data")
# =============================================================================
load_tot = pd.read_csv(r"load_tot.csv") #kWh
p_max_pu = pd.read_csv(r'p_max_pu.csv')

# technologies installed :
houses = 40
pv_installed = 1/4
storage_installed = 1/2


load_tot = load_tot.iloc[:,0:houses]
load_tot.set_axis(list(range(1,len(load_tot.columns)+1)),
                  axis = 1 ,inplace = True)

def add_date_ind(df):
    df["date"] = pd.date_range(start = pd.Timestamp("2021-01-01"),
                               periods=8760,freq="H",normalize=False)
    df.set_index("date",inplace = True)
    return df
add_date_ind(load_tot)
add_date_ind(p_max_pu)

# Build the network
# P2P network
n = pypsa.Network()
index=pd.date_range(start = pd.Timestamp("2021-01-01"),
                    periods=8760,freq="H",normalize=False)
n.set_snapshots(index)


# Add Carriers

n.add("Carrier", name='solar',color = "yellow")
n.add("Carrier", name='Gas',color = "Black",co2_emissions = 185)

# Add Bus

for b in range(0,len(load_tot.columns)+1):
    n.add("Bus",f"bus_{b}")
    
buses = list(n.buses.index)

# connect P2P network
#  connect the prosumers to the main bus 
line = 1
for bus in n.buses.index:
    if bus == "bus_0":
        continue
    else:
        n.add("Link",f"line_0_{line}",
              bus0 = "bus_0",
              bus1 = bus,
              p_nom = 100 , efficiency = 0.95)
        line+=1  

# connect prosumers with consumers 
s = 1       
for i in range(1,len(load_tot.columns)+1):
    for j in range(s+1,len(load_tot.columns)+1):
        n.add("Link",f"line_{i}_to_{j}",
                  bus0 = f'bus_{i}',
                  bus1 = f'bus_{j}',
                  p_nom = 500 ,efficiency = 0.95)
    s+=1

s = 1       
for i in range(1,len(load_tot.columns)+1):
    for j in range(s+1,len(load_tot.columns)+1):
        n.add("Link",f"line_{i}_from_{j}",
                  bus0 = f'bus_{j}',
                  bus1 = f'bus_{i}',
                  p_nom = 500 ,efficiency = 0.95)
    s+=1
    
# Add loads 

for bus in range(1,len(n.buses.index)):
    n.add("Load",
          f"load_{bus}",
          bus = f"bus_{bus}",
          p_set = (load_tot.iloc[:,bus-1])/1000)
        

"""
sum of load at any bus
"""
loads=pd.DataFrame(0,index=n.loads_t.p_set.index, columns=n.loads.bus.unique())
for bus in n.loads.bus.unique():
    if bus not in ['gen_load','bus_0']:
        loads[bus]=n.loads_t.p_set[n.loads.index[n.loads.bus==bus]].sum(axis=1)
# Add PV 


pv_gen = (load_tot.sum(axis= 0)*1.3).round(1) #MWh

for pv in range(1,int(len(load_tot.columns)*pv_installed)+1):
    n.add('Generator',
          f'PV_{pv}', 
          bus = f"bus_{pv}",
          p_nom = pv_gen[pv]/1000000,
          p_max_pu = p_max_pu.iloc[:,0],
          carrier = 'solar',
          marginal_cost = 120)

# Add Stores
pv_buses = [] 
pv_buses = n.generators.loc[n.generators.carrier == 'solar','bus']
for bus_storage in range(1,int((len(pv_buses))*storage_installed)+1):
    n.add('Store',
          name = f'storage_{bus_storage}',
          bus = f"bus_{bus_storage}",
          e_nom = 0.015)
    
# Add generator
n.add("Generator",name ="Main_Gen",bus = "bus_0",control='Slack',
      p_nom = 1,marginal_cost = 300,carrier = "Gas")

n.lopf(snapshots = n.snapshots,pyomo = True,solver_name = "gurobi")
os.chdir(r"/home/yalmousa1/Results/Final/26-02-2022")
n.export_to_netcdf('P2P_v1_Results.nc')

