"""

SUBJECT     : Structural Optimization Analysis of a Truss
NAME        : Michalis Lefkiou
DATE        : 17/04/2024
FILE NAME   : main.py

OVERVIEW    : This script performs structural analysis on a truss structure represented by nodes and members.
             It calculates the global stiffness matrix, global mass matrix, applies boundary conditions,
             and computes natural frequencies and mode shapes. Then it performs optimization for the cross-sectional
             area of each bar-truss.

DEPENDENCIES:
- numpy: For numerical operations.
- plotly.graph_objects: For visualizing the structure.
- helpers module: Contains helper functions for plotting and calculation of matrices.
- scipy.linalg.eigh: For computing eigenvalues and eigenvectors.

"""


# ======================================================================================================================
# --------------------------------------------------------IMPORTS-------------------------------------------------------
# ======================================================================================================================

import numpy as np
import plotly.graph_objects as go
from helpers import *
from scipy.linalg import eigh
from scipy.optimize import minimize
from functools import partial
from truss_structure import TrussStructure

# ======================================================================================================================
# -------------------------------------------------STRUCTURE GEOMETRY---------------------------------------------------
# ======================================================================================================================

L = 10                     # Parametrized Length of the Structure (m)
COORDS = np.array([
    [L, L, 0],  # Node 0
    [-L, L, 0],  # Node 1
    [-L, -L, 0],  # Node 2
    [L, -L, 0],  # Node 3

    [5 * L / 6, 5 * L / 6, 2 * L],  # Node 4
    [-5 * L / 6, 5 * L / 6, 2 * L],  # Node 5
    [-5 * L / 6, -5 * L / 6, 2 * L],  # Node 6
    [5 * L / 6, -5 * L / 6, 2 * L],  # Node 7

    [4 * L / 6, 4 * L / 6, 4 * L],  # Node 8
    [-4 * L / 6, 4 * L / 6, 4 * L],  # Node 9
    [-4 * L / 6, -4 * L / 6, 4 * L],  # Node 10
    [4 * L / 6, -4 * L / 6, 4 * L],  # Node 11

    [L / 2, L / 2, 6 * L],  # Node 12
    [-L / 2, L / 2, 6 * L],  # Node 13
    [-L / 2, -L / 2, 6 * L],  # Node 14
    [L / 2, -L / 2, 6 * L],  # Node 15
])   # Coordinates in (x,y,z)
MEMBERS = np.array([
    [0, 5],
    [1, 4],
    [1, 6],
    [2, 5],
    [2, 7],
    [3, 6],
    [3, 4],
    [0, 7],

    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],

    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],

    [4, 9],
    [5, 8],
    [5, 10],
    [6, 9],
    [6, 11],
    [7, 10],
    [7, 8],
    [4, 11],

    [4, 8],
    [5, 9],
    [6, 10],
    [7, 11],

    [8, 9],
    [9, 10],
    [10, 11],
    [11, 8],

    [8, 13],
    [9, 12],
    [9, 14],
    [10, 13],
    [10, 15],
    [11, 14],
    [11, 12],
    [8, 15],

    [8, 12],
    [9, 13],
    [10, 14],
    [11, 15],

    [12, 13],
    [13, 14],
    [14, 15],
    [15, 12]

])  # Member Connection Nodes [Node i, Node j]
DOF = 3                    # 3 axial-degrees of freedom (x,y,z)

# ======================================================================================================================
# ------------------------------------------PARAMETRIZATION OF THE PROBLEM----------------------------------------------
# ======================================================================================================================

E = 200E9                            # Young's Modulus (Pa)
A = 0.0001 * np.ones(len(MEMBERS))   # Cross-Sectional Area of the members (m^2)
rho = 7850                           # Density of the Material (kg/m^3)

# ======================================================================================================================
# -----------------------------------------------BOUNDARY CONDITIONS----------------------------------------------------
# ======================================================================================================================

CONSTRAINED_DOFS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# ======================================================================================================================
# --------------------------------------------SOLVE K * φi = ω^2 * M * φi = --------------------------------------------
# ======================================================================================================================

c1 = TrussStructure(coordinates=COORDS,
                    members=MEMBERS,
                    A=A,
                    E=E,
                    rho=rho,
                    num_dof=DOF,
                    constrained_dofs=CONSTRAINED_DOFS
                    )
EIGENVALUES, EIGENVECTORS = c1.calculate_eigenvalues()
c1.plot_structure()

# # Calculate the natural frequencies (omega)
NATURAL_FREQUENCIES = np.sqrt(EIGENVALUES)   # ωi = √(λi)
# ======================================================================================================================
# ---------------------------------------------OPTIMIZATION PROBLEM-----------------------------------------------------
# ======================================================================================================================

# MASS = objective_function(A=A, rho=rho, coordinates=COORDS, members=MEMBERS)
#
# A_LOWER_BOUND = 0.01
# A_UPPER_BOUND = 1.0
#
# A_INITIAL_GUESS = np.random.uniform(A_LOWER_BOUND, A_UPPER_BOUND, len(MEMBERS))
#
# OBJECTIVE_FUNCTION = partial(objective_function, rho=rho, coordinates=COORDS, members=MEMBERS)
#
# # Define the objective function to be minimized, taking only the design variables as input
# OBJ_TO_MIN = lambda A: OBJECTIVE_FUNCTION(A)
#
# # Minimize the objective function
# RESULT = minimize(OBJ_TO_MIN, A_INITIAL_GUESS, method='nelder-mead', bounds=[(A_LOWER_BOUND, A_UPPER_BOUND)] * len(MEMBERS))
#
# # Extract the optimal design variables
# OPTIMAL_A = RESULT.x
#
# # Evaluate the total mass using the optimal design variables
# OPTIMAL_MASS = OBJECTIVE_FUNCTION(OPTIMAL_A)
#
# print("Optimal design variables (A):", OPTIMAL_A)
# print("Total mass with optimal design variables:", OPTIMAL_MASS)


# ======================================================================================================================
# --------------------------------------------------PLOT STRUCTURE------------------------------------------------------
# ======================================================================================================================

# plot_structure(
#     COORDS, MEMBERS,
#     title='Offshore Platform Visualization',
#     x_label='x (m)',
#     y_label='y (m)',
#     z_label='z (m)',
#     node_color='red',
#     node_size=4,
#     member_color='black',
#     member_width=2,
#     style='plotly_white'
# )

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # Select the mode shape to plot (change index as needed)
# mode_index = 2
#
# # Extract mode shape for the selected mode
# mode_shape = EIGENVECTORS[:, mode_index]
#
# # Scale mode shape for visualization
# scaled_mode_shape = mode_shape * 100.0  # Adjust scaling factor as needed
#
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot mode shape as lines between nodes
# for member in MEMBERS:
#     start_node = COORDS[member[0]]
#     end_node = COORDS[member[1]]
#     ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], [start_node[2], end_node[2]], color='blue')
#
# # Plot scaled mode shape as arrows from nodes
# for i, coord in enumerate(COORDS):
#     x, y, z = coord
#     if len(scaled_mode_shape) >= (i + 1) * DOF:
#         dx, dy, dz = scaled_mode_shape[i * DOF: (i + 1) * DOF]
#         ax.quiver(x, y, z, dx, dy, dz, color='green')
#
# # Set plot labels
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title(f'Mode Shape {mode_index + 1}')
#
# # Show plot
# plt.show()

