import numpy as np


class Object:
    def __init__(self, name: str, n_samples=200):
        """
        Get a PyVista object by name.
        """
        self.name = name
        self.n_samples = n_samples
        if self.name == "arrow":
            self.data = np.array(
                [
                    [0, 0],
                    [0.25, 0],
                    [0.25, 0.25],
                    [0.25, 0.5],
                    [0.5, 0.5],
                    [0.25, 0.75],
                    [0, 1],
                    [-0.25, 0.75],
                    [-0.5, 0.5],
                    [-0.25, 0.5],
                    [-0.25, 0.25],
                    [-0.25, 0],
                    [0, 0],
                ],
                dtype=np.float32,
            )

        elif self.name == "half_arrow":
            self.data = np.array(
                [
                    [0, 0],
                    [0.25, 0],
                    [0.25, 0.25],
                    [0.25, 0.5],
                    [0.5, 0.5],
                    [0.25, 0.75],
                    [0, 1],
                ],
                dtype=np.float32,
            )

        elif self.name == "pv_half_arrow_4pts":
            self.data = np.array(
                [[0, 0, 0], [1.0, 0, 0], [0.8, 0.15, 0.15], [0.8, -0.15, 0.15]],
                dtype=np.float32,
            )
        elif self.name == "irreg_tet":
            self.data = np.array(
                [[0, 0, 0], [1.0, 0, 0], [0.5, 1, 0.2], [0.3, 0.4, 1.1]],
                dtype=np.float32,
            )
        elif self.name == "irreg_tet_4d":
            # 4D version of irreg_tet with zero padding in the 4th dimension
            self.data = np.array(
                [[0, 0, 0, 1], [1.0, 0, 0, 1], [0.5, 1, 0.2, 1], [0.3, 0.4, 1.1, 1]],
                dtype=np.float32,
            )
        elif self.name == "irreg_hex":
            # Irregular hexahedron (8 vertices)
            # Create a non-regular hexahedron with 8 vertices
            # Bottom face: 4 vertices
            # Top face: 4 vertices (shifted and rotated)
            self.data = np.array(
                [
                    # Bottom face
                    [-0.5, -0.5, -0.5],  # 0
                    [0.6, -0.4, -0.5],   # 1
                    [0.5, 0.7, -0.3],    # 2
                    [-0.4, 0.5, -0.4],   # 3
                    # Top face
                    [-0.3, -0.6, 0.6],   # 4
                    [0.7, -0.3, 0.5],    # 5
                    [0.4, 0.8, 0.7],     # 6
                    [-0.5, 0.4, 0.6],    # 7
                ],
                dtype=np.float32,
            )
        elif self.name == "triangular_prism":
            # Triangular prism (5 faces, 6 vertices)
            # Bottom triangle: 3 vertices
            # Top triangle: 3 vertices (shifted along z-axis)
            self.data = np.array(
                [
                    # Bottom triangle
                    [0.0, 0.0, -0.5],      # 0
                    [0.8, 0.0, -0.4],     # 1
                    [0.4, 0.7, -0.3],     # 2
                    # Top triangle
                    [0.1, 0.1, 0.5],      # 3
                    [0.7, 0.1, 0.6],      # 4
                    [0.3, 0.8, 0.4],      # 5
                ],
                dtype=np.float32,
            )
        elif self.name == "octahedron":
            # Octahedron (8 faces, 6 vertices)
            # Regular octahedron: 6 vertices forming two pyramids
            self.data = np.array(
                [
                    [0.0, 0.0, 0.8],      # 0 - Top
                    [0.6, 0.0, 0.0],      # 1 - Front
                    [0.0, 0.6, 0.0],      # 2 - Right
                    [-0.6, 0.0, 0.0],     # 3 - Back
                    [0.0, -0.6, 0.0],     # 4 - Left
                    [0.0, 0.0, -0.8],     # 5 - Bottom
                ],
                dtype=np.float32,
            )
        elif self.name == "dodecahedron":
            # Dodecahedron (12 faces, 20 vertices)
            # Regular dodecahedron vertices (simplified irregular version)
            # Using a more manageable irregular dodecahedron
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            a = 0.5
            b = 0.5 / phi
            c = 0.5 * phi
            
            self.data = np.array(
                [
                    # First set of 8 vertices (cube-like)
                    [a, a, a], [a, a, -a], [a, -a, a], [a, -a, -a],
                    [-a, a, a], [-a, a, -a], [-a, -a, a], [-a, -a, -a],
                    # Additional 12 vertices (simplified)
                    [0, b, c], [0, b, -c], [0, -b, c], [0, -b, -c],
                    [b, c, 0], [b, -c, 0], [-b, c, 0], [-b, -c, 0],
                    [c, 0, b], [c, 0, -b], [-c, 0, b], [-c, 0, -b],
                ],
                dtype=np.float32,
            )
            # Scale down to match other objects
            self.data = self.data * 0.6
        elif self.name == "triangle":
            # Triangle with 3 vertices in 2D, centroid at origin
            # Represented as 6D vector: [v1_x, v1_y, v2_x, v2_y, v3_x, v3_y]
            # Create an equilateral triangle centered at origin
            v1 = np.array([0.5, 0.0], dtype=np.float32)
            v2 = np.array([-0.25, 0.433], dtype=np.float32)
            v3 = np.array([-0.25, -0.433], dtype=np.float32)
            # Center at origin
            centroid = (v1 + v2 + v3) / 3.0
            v1 = v1 - centroid
            v2 = v2 - centroid
            v3 = v3 - centroid
            # Store as [1, 6] for ObjectTransformDistribution
            # The distribution will return [B, 1, 6], which we'll flatten to [B, 6] in the dataset
            self.data = np.concatenate([v1, v2, v3], axis=0).reshape(1, 6)
        elif self.name == "identity":
            # Identity point cloud: 3 points forming a 3x3 identity matrix
            # Points: [1,0,0], [0,1,0], [0,0,1]
            self.data = np.array(
                [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            )
        elif self.name == "identity_4d":
            # 4D version of identity point cloud with zero padding in the 4th dimension
            # Points: [1,0,0,0], [0,1,0,0], [0,0,1,0]
            self.data = np.array(
                [[1., 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                dtype=np.float32,
            )
        elif self.name == "square":
            # Square with 8 points: 4 corners + 4 edge midpoints, in_dim = 16
            h = 0.5
            self.data = np.array(
                [
                    [-h, -h],   # bottom-left corner
                    [ 0, -h],   # bottom midpoint
                    [ h, -h],   # bottom-right corner
                    [ h,  0],   # right midpoint
                    [ h,  h],   # top-right corner
                    [ 0,  h],   # top midpoint
                    [-h,  h],   # top-left corner
                    [-h,  0],   # left midpoint
                ],
                dtype=np.float32,
            )
        elif self.name == "rectangle":
            # Rectangle (width=1.0, height=0.5) with 8 points:
            # 4 corners + 4 edge midpoints, in_dim = 16
            w, h = 0.5, 0.25   # half-width, half-height
            self.data = np.array(
                [
                    [-w, -h],   # bottom-left corner
                    [ 0, -h],   # bottom midpoint
                    [ w, -h],   # bottom-right corner
                    [ w,  0],   # right midpoint
                    [ w,  h],   # top-right corner
                    [ 0,  h],   # top midpoint
                    [-w,  h],   # top-left corner
                    [-w,  0],   # left midpoint
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Object '{self.name}' is not supported.")
