from typing import Tuple
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli


class LayeredToricPauli(RotatedPlanar3DPauli):

    def vertex_on_defect_boundary(self, L_x, L_y, x, y):
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

        defect_x_boundary, defect_y_boundary = self.vertex_on_defect_boundary(
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
        # TODO deal with boundary conditions and defects
        else:

            neighbours = []

            # Those in-plane with unrotated y (normal to unrotated x)
            if (x - y) % 4 == 2:

                # y boundary.
                if y == 1:

                    # Neighbouring vertical edge in bulk.
                    neighbours.append((x + 1, y + 1, z))

                    # Interpolate across compatible boundary.
                    if L_y % 2 == 0:
                        neighbours.append((x - 1, 2*L_y, z))

                    # Twist across incompatible boundary.
                    else:
                        neighbours.append((x + 1, 2*L_y, z))

                # x boundary.
                elif x == 1:

                    # Neighbouring vertical edge in bulk.
                    neighbours.append((x + 1, y + 1, z))

                    # Interpolate across compatible boundary.
                    if L_x % 2 == 0:
                        neighbours.append((2*L_x, y - 1, z))

                    # Twist across incompatible boundary.
                    else:
                        neighbours.append((2*L_x, y + 1, z))

                # Typical case in the bulk.
                else:
                    neighbours.append((x + 1, y + 1, z))
                    neighbours.append((x - 1, y - 1, z))

            # Those in-plane with unrotated x (normal to unrotated y)
            else:

                # Corner case.
                if (x, y) == (1, 1):
                    if L_x % 2 == 0 and L_y % 2 == 0:
                        neighbours.append((2, 2*L_y, z))
                        neighbours.append((2*L_x, 2, z))
                    elif L_x % 2 == 0 and L_y % 2 == 1:
                        neighbours.append((2*L_x, 2*L_y, z))
                        neighbours.append((2*L_x, 2, z))
                    elif L_x % 2 == 1 and L_x % 2 == 0:
                        neighbours.append((2, 2*L_y, z))
                        neighbours.append((2*L_x, 2*L_y, z))
                    else:
                        pass

                # y boundary.
                elif y == 1:
                    pass

                # x boundary.
                elif x == 1:
                    pass

                # Typical case in the bulk.
                else:
                    neighbours.append((x + 1, y - 1, z))
                    neighbours.append((x - 1, y + 1, z))

            # Edges top and bottom.
            if z + 1 <= 2*L_z + 1:
                neighbours.append((x, y, z + 1))
            if x - 1 >= 1:
                neighbours.append((x, y, z - 1))

        for edge in neighbours:
            self.site(operator, edge)
