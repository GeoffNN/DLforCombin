"""
We use the Full Whiskas Model Python Formulation for the PuLP Modeller
We make the cost of different ingredients vary and observe the corresponding optimal quantity for ingredients
Authors: Antony Phillips, Dr Stuart Mitchell  2007

Modifications: Bertrand Travacca 2017
"""

# Import PuLP modeler functions
from pulp import *
import random as rd
import numpy as np

# Creates a list of the Ingredients
Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']


# A dictionary of the costs of each of the Ingredients is created
reference_costs = {'CHICKEN': 0.013, 
         'BEEF': 0.008, 
         'MUTTON': 0.010, 
         'RICE': 0.002, 
         'WHEAT': 0.005, 
         'GEL': 0.001}

# A dictionary of the protein percent in each of the Ingredients is created
proteinPercent = {'CHICKEN': 0.100, 
                  'BEEF': 0.200, 
                  'MUTTON': 0.150, 
                  'RICE': 0.000, 
                  'WHEAT': 0.040, 
                  'GEL': 0.000}

# A dictionary of the fat percent in each of the Ingredients is created
fatPercent = {'CHICKEN': 0.080, 
              'BEEF': 0.100, 
              'MUTTON': 0.110, 
              'RICE': 0.010, 
              'WHEAT': 0.010, 
              'GEL': 0.000}

# A dictionary of the fibre percent in each of the Ingredients is created
fibrePercent = {'CHICKEN': 0.001, 
                'BEEF': 0.005, 
                'MUTTON': 0.003, 
                'RICE': 0.100, 
                'WHEAT': 0.150, 
                'GEL': 0.000}

# A dictionary of the salt percent in each of the Ingredients is created
saltPercent = {'CHICKEN': 0.002, 
               'BEEF': 0.005, 
               'MUTTON': 0.007, 
               'RICE': 0.002, 
               'WHEAT': 0.008, 
               'GEL': 0.000}
N_data=1000
optimum_sol=np.zeros((6, N_data))
costs_values=np.zeros((6, N_data))

for k in range(N_data):
    # A dictionary of the costs of each of the Ingredients is created, it takes the reference cost and draws u.a.r
    # independently for each ingredient a cost within 50% of the reference
    
    costs_values[0,k]=0.013*(0.5+rd.random())
    costs_values[1,k]=0.008*(0.5+rd.random())
    costs_values[2,k]=0.010*(0.5+rd.random())
    costs_values[3,k]=0.002*(0.5+rd.random())
    costs_values[4,k]=0.005*(0.5+rd.random())
    costs_values[5,k]=0.001*(0.5+rd.random())
    
    costs = {'CHICKEN': costs_values[0,k], 
             'BEEF':costs_values[1,k] , 
             'MUTTON': costs_values[2,k], 
             'RICE':costs_values[3,k] , 
             'WHEAT': costs_values[4,k], 
             'GEL': costs_values[5,k]}

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("The Whiskas Problem", LpMinimize)

    # A dictionary called 'ingredient_vars' is created to contain the referenced Variables
    ingredient_vars = LpVariable.dicts("Ingr",Ingredients,0)

    # The objective function is added to 'prob' first
    prob += lpSum([costs[i]*ingredient_vars[i] for i in Ingredients]), "Total Cost of Ingredients per can"

    # The five constraints are added to 'prob'
    prob += lpSum([ingredient_vars[i] for i in Ingredients]) == 100, "PercentagesSum"
    prob += lpSum([proteinPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 8.0, "ProteinRequirement"
    prob += lpSum([fatPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 6.0, "FatRequirement"
    prob += lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0, "FibreRequirement"
    prob += lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4, "SaltRequirement"

    # The problem data is written to an .lp file
    prob.writeLP("WhiskasModel2.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    #print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    l=0
    for v in prob.variables():
        optimum_sol[l,k]=v.varValue
        l=l+1
