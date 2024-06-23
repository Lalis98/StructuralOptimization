"""

SUBJECT     : Structural Optimization Analysis of a Truss for Natural Frequencies
NAME        : Michalis Lefkiou
DATE        : 17/04/2024
FILE NAME   : main.py

OVERVIEW    : This script performs structural analysis on a structure represented by nodes and members.
             It calculates the global stiffness matrix, global mass matrix, applies boundary conditions,
             and computes natural frequencies and mode shapes. Then it performs optimization for the cross-sectional
             area of each bar-structure.

DEPENDENCIES:
- numpy                   : For numerical operations.
- scipy.optimize.minimize : For performing numerical optimization.
- plot_iterations         : For visualizing optimization progress.
- plot_frequencies        : For visualizing natural frequencies.
- areas_to_excel          : For exporting optimized areas to Excel.
- TrussStructure          : Defines the truss structure and methods for analysis.

"""


def main():
    # =========================================================================
    #                                 IMPORTS
    # =========================================================================

    import numpy as np
    from scipy.optimize import minimize
    from plotting_functions import plot_iterations, plot_frequencies, areas_to_excel
    from truss_structure import TrussStructure

    # =========================================================================
    #                          OPTIMIZATION FUNCTIONS
    # =========================================================================

    OBJECTIVE_VALUES = []  # Define a list to store objective function values

    def penalty_function(natural_frequencies, frequency_constraints):
        penalty = 0
        for i in range(len(natural_frequencies)):
            penalty += np.maximum(0, 1 - (natural_frequencies[i] / frequency_constraints[i]))
        return penalty

    def calculate_total_mass(A_optimization, rho, coordinates, members):
        total_mass = 0
        for c, connection in enumerate(members):
            node_i_index, node_j_index = connection
            length = np.linalg.norm(coordinates[node_i_index] - coordinates[node_j_index])
            total_mass += rho * A_optimization[c] * length
        return total_mass

    def objective_function(A_optimization):
        _, _, natural_frequencies = structure.calculate_eigenvalues(A_optimization)
        total_penalty_value = penalty_function(natural_frequencies, FREQUENCY_CONSTRAINTS)
        total_mass = calculate_total_mass(A_optimization, RHO, COORDS, MEMBERS)
        return total_mass * (1 + C1 * total_penalty_value) ** C2

    def callback_function(xk):
        obj_value = objective_function(xk)
        OBJECTIVE_VALUES.append(obj_value)
        print("Iteration:", len(OBJECTIVE_VALUES), "| Objective Function Value:", obj_value)

    # =========================================================================
    #                              STRUCTURE GEOMETRY
    # =========================================================================

    L = 10                      # Parametrized Length of the Structure (m)
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
    ])    # Coordinates in (x,y,z)
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

    ])   # Member Connection Nodes [Node i, Node j]
    DOF = 3                     # 3D - Space Bar Truss = 3 axial-degrees of freedom (x,y,z)

    # =========================================================================
    #                                  ADDED MASS
    # =========================================================================

    ADDED_MASS_INDICES = np.array([42, 43, 44, 45])      # Indices for nodes that mass will be added
    ADDED_MASS = np.array([1_000, 1_000, 1_000, 1_000])  # Mass that will be added in the corresponding node index

    # =========================================================================
    #                       PARAMETRIZATION OF THE PROBLEM
    # =========================================================================

    A_LOWER_BOUND = 0.001              # Minimum Cross Sectional Area (m^2)
    A_UPPER_BOUND = 0.002              # Maximum Cross Sectional Area (m^2)
    E = 200E9                          # Young's Modulus (Pa)
    A = 0.001 * np.ones(len(MEMBERS))  # Cross-Sectional Area of the members (m^2)
    RHO = 7_850                        # Density of the Material (kg/m^3)
    print("Initial Cross-Sectional Areas:", A)

    # =========================================================================
    #                            BOUNDARY CONDITIONS
    # =========================================================================

    CONSTRAINED_DOFS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # DOFs that are fixed

    # =========================================================================
    #                         SOLVE K * v = ω^2 * M * v
    # =========================================================================

    # Initialize object "structure"
    structure = TrussStructure(
        coordinates=COORDS,
        members=MEMBERS,
        A=A,
        E=E,
        rho=RHO,
        num_dof=DOF,
        constrained_dofs=CONSTRAINED_DOFS,
        added_mass_indices=ADDED_MASS_INDICES,
        added_mass=ADDED_MASS
    )

    # eigenvalues, eigenvectors and natural frequencies
    EIGENVALUES, EIGENVECTORS, NATURAL_FREQUENCIES = structure.calculate_eigenvalues(A)
    print("Initial Natural Frequencies (Hz):", NATURAL_FREQUENCIES)

    # =========================================================================
    #                             OPTIMIZATION PROBLEM
    # =========================================================================

    FREQUENCY_CONSTRAINTS = np.ones(len(NATURAL_FREQUENCIES))  # Frequency Constraints [ω1, ω2, ..., ωN]
    FREQUENCY_CONSTRAINTS[0:3] = np.array([2.5, 4, 7])         # Assign the preferred frequency constraints
    BOUNDS = [(A_LOWER_BOUND, A_UPPER_BOUND)] * len(A)         # Bounds for Cross Sectional Area [Amin, Amax]
    C1 = 100.0                                                 # Coefficient used in penalisation of objective function
    C2 = 2.0                                                   # Coefficient used in penalisation of objective function

    # Run the optimization
    RESULT = minimize(
        objective_function,
        x0=A,
        bounds=BOUNDS,
        method='Nelder-Mead',
        callback=callback_function
    )

    OPTIMIZED_A = RESULT.x  # Extract the optimized cross-sectional areas and eigenvalues
    _, _, OPTIMIZED_FREQUENCIES = structure.calculate_eigenvalues(OPTIMIZED_A)

    # Print the optimized cross-sectional areas and eigenvalues
    print("Optimized Cross-Sectional Areas:", OPTIMIZED_A)
    print("Optimized Frequencies:", OPTIMIZED_FREQUENCIES)

    # =========================================================================
    #                                 PLOTTING
    # =========================================================================

    ITERATIONS = np.arange(1, len(OBJECTIVE_VALUES) + 1)   # Iterations
    FREQUENCIES_HZ = OPTIMIZED_FREQUENCIES[:6]             # First six natural frequencies

    plot_iterations(ITERATIONS, OBJECTIVE_VALUES, max_frames=25)
    plot_frequencies(FREQUENCIES_HZ)
    areas_to_excel(MEMBERS, OPTIMIZED_A, 'optimized_areas.xlsx')


if __name__ == "__main__":
    main()
