from typing import Tuple
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli


class LayeredToricPauli(RotatedPlanar3DPauli):

    def on_defect_boundary(self, L_x, L_y, x, y):
        """Determine whether or not to defect each boundary."""
        defect_x_boundary = False
        defect_y_boundary = False
        if L_x % 2 == 1 and x == 2*L_x:
            defect_x_boundary = True
        if L_y % 2 == 1 and y == 2*L_y:
            defect_y_boundary = True
        return defect_x_boundary, defect_y_boundary

    def vertex(self, operator: str, location: Tuple[int, int, int]):
        L_x, L_y, L_z = self.code.size
        x, y, z = location
        neighbours = []

        # Edges along unrotated x with periodic boundaries.
        neighbours += [
            ((x + 1) % (2*L_x), y - 1, z),
            (x - 1, (y + 1) % (2*L_y), z)
        ]

        # Edges along unrotated y with periodic boundaries.
        neighbours += [
            ((x + 1) % (2*L_x), (y + 1) % (2*L_y), z),
            (x - 1, y - 1, z)
        ]

        # Vertical edges along unrotated z, with smooth boundaries.
        if z + 1 <= 2*L_z + 1:
            neighbours.append((x, y, z + 1))
        if z - 1 >= 1:
            neighbours.append((x, y, z - 1))

        defect_x_boundary, defect_y_boundary = self.on_defect_boundary(
            L_x, L_y, x, y
        )
        if defect_x_boundary or defect_y_boundary:
            defect_operator = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}[operator]

        for edge in neighbours:
            defect_x_on_edge = defect_x_boundary and edge[0] == 1
            defect_y_on_edge = defect_y_boundary and edge[1] == 1
            if defect_x_on_edge != defect_y_on_edge:
                self.site(defect_operator, edge)
            else:
                self.site(operator, edge)

    def face(self, operator: str, location: Tuple[int, int, int]):
        L_x, L_y, L_z = self.code.size
        x, y, z = location

        # Horizontal plane faces.
        if z % 2 == 1:
            neighbours = [
                ((x + 1) % (2*L_x), (y + 1) % (2*L_y), z),
                ((x + 1) % (2*L_x), y - 1, z),
                (x - 1, (y + 1) % (2*L_y), z),
                (x - 1, y - 1, z),
            ]

        # Vertical faces.
        else:

            neighbours = []

            # Horizontal plane surrounding edges.
            possible_sites = [
                (x + 1, y + 1, z), (x - 1, y - 1, z),
                (x + 1, y - 1, z), (x - 1, y + 1, z),
                (x, y, z + 1), (x, y, z - 1)
            ]

            # Apply periodic boundary conditions if compatible lattice.
            if L_x % 2 == 0:
                for i_site, site in enumerate(possible_sites):
                    if site[0] == 0:
                        possible_sites[i_site] = (2*L_x, site[1], site[2])
                    elif site[0] == 2*L_x + 1:
                        possible_sites[i_site] = (1, site[1], site[2])

            # Otherwise no sites if odd boundary.
            elif x == 1:
                possible_sites = []

            # Apply periodic boundary conditions if compatible lattice.
            if L_y % 2 == 0:
                for i_site, site in enumerate(possible_sites):
                    if site[1] == 0:
                        possible_sites[i_site] = (site[0], 2*L_y, site[2])
                    elif site[1] == 2*L_y + 1:
                        possible_sites[i_site] = (site[0], 1, site[2])

            # Otherwise no sites if odd boundary.
            elif y == 1:
                possible_sites = []

            for site in possible_sites:
                if site in self.code.qubit_index:
                    neighbours.append(site)

        defect_x_boundary, defect_y_boundary = self.on_defect_boundary(
            L_x, L_y, x, y
        )
        if defect_x_boundary or defect_y_boundary:
            defect_operator = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}[operator]

        for edge in neighbours:
            defect_x_on_edge = defect_x_boundary and edge[0] == 1
            defect_y_on_edge = defect_y_boundary and edge[1] == 1
            if defect_x_on_edge != defect_y_on_edge:
                self.site(defect_operator, edge)
            else:
                self.site(operator, edge)
