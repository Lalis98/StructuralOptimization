import numpy as np
from scipy.linalg import eigh
import plotly.graph_objects as go

class TrussStructure:
    def __init__(self, coordinates, members, A, E, rho, num_dof, constrained_dofs):

        self.coordinates = coordinates
        self.members = members
        self.A = A
        self.E = E
        self.rho = rho
        self.num_dof = num_dof
        self.constrained_dofs = constrained_dofs
        # self.free_dofs = free_dofs

        # Number of Degrees of Freedom
        self.nodes_dof = np.arange(len(coordinates) * num_dof)

        # Number of members' Degrees of Freedom
        self.members_dof = []
        for i in self.members:
            member_dof = [
                i[0] * self.num_dof, i[0] * self.num_dof + 1, i[0] * self.num_dof + 2,
                i[1] * self.num_dof, i[1] * self.num_dof + 1, i[1] * self.num_dof + 2
            ]
            self.members_dof.append(member_dof)

        self.members_dof = np.array(self.members_dof)

        self.free_dofs = np.setdiff1d(self.nodes_dof, constrained_dofs)

    @staticmethod
    def _calculate_length(node_i, node_j):
        """
        Calculate the 3D distance between two points.

        Parameters:
        node_i (numpy.ndarray): Coordinates of the first node (x, y, z).
        node_j (numpy.ndarray): Coordinates of the second node (x, y, z).

        Returns:
        float: The distance between the two nodes.
        """

        xi, yi, zi = node_i
        xj, yj, zj = node_j

        length = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)

        return length

    @staticmethod
    def _calculate_local_stiffness_matrix(node_i, node_j, E, Ae):
        """
        Calculate the local stiffness matrix for a 3D truss element.

        Args:
            node_i (numpy.ndarray): Coordinates (x, y, z) of the first node.
            node_j (numpy.ndarray): Coordinates (x, y, z) of the second node.
            E (float): Modulus of elasticity.
            Ae (float): Cross-sectional area.

        Returns:
            numpy.ndarray: 6x6 local stiffness matrix.
        """

        L = TrussStructure._calculate_length(node_i, node_j)

        R = TrussStructure._calculate_rotation_matrix(node_i, node_j)

        k = E * Ae / L * np.array([
            [1, -1],
            [-1, 1]
        ])

        K_local = np.dot(R.T, np.dot(k, R))

        return K_local

    @staticmethod
    def _calculate_global_stiffness_matrix(dof, coordinates, members, members_dof, E, A):
        """
        Calculate the global stiffness matrix for a truss structure.

        Args:
            dof (int): Degrees of freedom per node.
            coordinates (numpy.ndarray): Array of node coordinates.
            members (numpy.ndarray): Numpy array representing member connections.
            members_dof (numpy.ndarray): Numpy array representing member degrees of freedom.
            E (float): Modulus of elasticity.
            A (numpy.ndarray): Array of cross-sectional areas for each member.

        Returns:
            numpy.ndarray: Global stiffness matrix.
        """

        K_global = np.zeros((len(coordinates) * dof, len(coordinates) * dof))  # Stiffness Matrix

        for c, connection in enumerate(members):
            node_i_index, node_j_index = connection  # node 1 & node 2

            selection = np.ix_(members_dof[c], members_dof[c])  # Selection of the Connectivities

            # Stiffness Matrix of the element
            Ke = TrussStructure._calculate_local_stiffness_matrix(coordinates[node_i_index],
                                                                  coordinates[node_j_index], E, A[c])

            K_global[selection] += Ke  # Adding element's Stiffness Matrix to global Stiffness Matrix

        return K_global

    @staticmethod
    def _calculate_rotation_matrix(node_i, node_j):
        """
        Calculate the rotation matrix for a truss element based on the coordinates
        of its endpoints.

        Args:
            node_i (numpy.ndarray): Coordinates (x, y, z) of the first node.
            node_j (numpy.ndarray): Coordinates (x, y, z) of the second node.

        Returns:
            numpy.ndarray: The 2x6 rotation matrix.
        """

        xi, yi, zi = node_i
        xj, yj, zj = node_j

        L = TrussStructure._calculate_length(node_i, node_j)

        cx = (xj - xi) / L
        cy = (yj - yi) / L
        cz = (zj - zi) / L

        R = np.array([
            [cx, cy, cz, 0, 0, 0],
            [0, 0, 0, cx, cy, cz]
        ])

        return R

    @staticmethod
    def _calculate_local_mass_matrix(node_i, node_j, rho, Ae):
        """
        Calculate the consistent mass matrix for a 3D truss element based on
        the coordinates of its endpoints and its material properties.

        Args:
            node_i (numpy.ndarray): Coordinates (x, y, z) of the first node.
            node_j (numpy.ndarray): Coordinates (x, y, z) of the second node.
            rho (float): Density of the material of the element.
            Ae (float): Cross-sectional area of the element.

        Returns:
            numpy.ndarray: The 6x6 consistent mass matrix for the truss element.
        """

        L = TrussStructure._calculate_length(node_i, node_j)

        R = TrussStructure._calculate_rotation_matrix(node_i, node_j)

        m = (rho * Ae * L / 6) * np.array([
            [2, 1],
            [1, 2],
        ])

        M_local = np.dot(R.T, np.dot(m, R))

        return M_local

    @staticmethod
    def _calculate_global_mass_matrix(dof, coordinates, members, members_dof, rho, A):
        """
        Calculate the global consistent mass matrix for a 3D truss structure.

        Args:
            dof (int): Degrees of freedom per node.
            coordinates (numpy.ndarray): Array of node coordinates.
            members (numpy.ndarray): List of member connections.
            members_dof (numpy.ndarray): List of degrees of freedom for each member.
            rho (float): Density of the material of the truss elements.
            A (numpy.ndarray): List of cross-sectional areas of the truss elements.

        Returns:
            numpy.ndarray: The global consistent mass matrix for the truss structure.
        """

        M_global = np.zeros((len(coordinates) * dof, len(coordinates) * dof))  # Stiffness Matrix

        for c, connection in enumerate(members):
            node_i_index, node_j_index = connection  # node 1 & node 2

            selection = np.ix_(members_dof[c], members_dof[c])  # Selection of the Connections

            Me = TrussStructure._calculate_local_mass_matrix(coordinates[node_i_index],
                                                             coordinates[node_j_index], rho, A[c])

            M_global[selection] += Me  # Adding element's Mass Matrix to global Stiffness Matrix

        return M_global

    @staticmethod
    def _split_matrix(matrix, constrained_dofs, free_dofs):
        """
        Split the given matrix into constrained and free matrices.

        Parameters:
            matrix (numpy.ndarray): The matrix to be split.
            constrained_dofs (numpy.ndarray): List of indices of constrained degrees of freedom.
            free_dofs (numpy.ndarray): List of indices of free degrees of freedom.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray:
            The split matrices Kcc, Kcf, Kfc, Kff (or Mcc, Mcf, Mfc, Mff).
        """


        Matrix_cc = matrix[np.ix_(constrained_dofs, constrained_dofs)]
        Matrix_cf = matrix[np.ix_(constrained_dofs, free_dofs)]
        Matrix_fc = matrix[np.ix_(free_dofs, constrained_dofs)]
        Matrix_ff = matrix[np.ix_(free_dofs, free_dofs)]

        return Matrix_cc, Matrix_cf, Matrix_fc, Matrix_ff

    def calculate_eigenvalues(self):

        K = TrussStructure._calculate_global_stiffness_matrix(self.num_dof, self.coordinates, self.members,
                                                              self.members_dof, self.E, self.A)

        M = TrussStructure._calculate_global_mass_matrix(self.num_dof, coordinates=self.coordinates,
                                                         members=self.members, members_dof=self.members_dof,
                                                         rho=self.rho, A=self.A)

        (Kcc, Kcf, Kfc, Kff) = TrussStructure._split_matrix(matrix=K,
                                                            constrained_dofs=self.constrained_dofs,
                                                            free_dofs=self.free_dofs)
        (Mcc, Mcf, Mfc, Mff) = TrussStructure._split_matrix(matrix=M,
                                                            constrained_dofs=self.constrained_dofs,
                                                            free_dofs=self.free_dofs)

        EIGENVALUES, EIGENVECTORS = eigh(Kff, Mff)

        # # Calculate the natural frequencies (omega)
        # NATURAL_FREQUENCIES = np.sqrt(EIGENVALUES)  # ωi = √(λi)

        return EIGENVALUES, EIGENVECTORS

    def plot_structure(self, title='Structure Visualization', x_label='X axis', y_label='Y axis', z_label='Z axis', node_color='red', node_size=4, member_color='black', member_width=1, style=None):
        """
        Plot 3D structure with nodes and members.

        Parameters:
            title (str): Title of the plot.
            x_label (str): Label for the X axis.
            y_label (str): Label for the Y axis.
            z_label (str): Label for the Z axis.
            node_color (str): Color of nodes.
            node_size (int): Size of nodes.
            member_color (str): Color of members.
            member_width (int): Width of members.
            style (str): Name of the predefined style to apply (e.g., 'plotly', 'seaborn', 'ggplot2', etc.).
        """

        x_coords = self.coordinates[:, 0]  # x-coordinates of all nodes
        y_coords = self.coordinates[:, 1]  # y-coordinates of all nodes
        z_coords = self.coordinates[:, 2]  # z-coordinates of all nodes

        # Plot the nodes
        nodes_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=0.8
            ),
            name='Nodes',
            showlegend=False  # Turn off legend
        )

        # Plot the members
        lines = []
        for m in self.members:
            x_vals = [x_coords[m[0]], x_coords[m[1]], None]
            y_vals = [y_coords[m[0]], y_coords[m[1]], None]
            z_vals = [z_coords[m[0]], z_coords[m[1]], None]
            line = go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines',
                line=dict(color=member_color, width=member_width),
                name='Member',
                showlegend=False
            )
            lines.append(line)

        # Plot the layout
        fig = go.Figure(data=[nodes_trace, *lines],
                        layout=go.Layout(
                            title=title,
                            scene=dict(
                                xaxis=dict(title=x_label),
                                yaxis=dict(title=y_label),
                                zaxis=dict(title=z_label)
                            ),
                            template=style,  # Apply predefined style
                            plot_bgcolor='white'  # Change background to white
                        ))

        fig.show()
