import numpy as np


def find_boundary(mesh_elements):
    boundary_points_ = set()
    boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
    boundary_triangles_ = np.zeros(shape=(0, 3), dtype=np.int32)
    if mesh_elements.shape[1] == 3:
        # 2D triangle mesh
        edges = set()
        for [i, j, k] in mesh_elements:
            edges.add((i, j))
            edges.add((j, k))
            edges.add((k, i))
        for [i, j, k] in mesh_elements:
            if (j, i) not in edges:
                boundary_points_.update([j, i])
                boundary_edges_ = np.vstack((boundary_edges_, [j, i]))
            if (k, j) not in edges:
                boundary_points_.update([k, j])
                boundary_edges_ = np.vstack((boundary_edges_, [k, j]))
            if (i, k) not in edges:
                boundary_points_.update([i, k])
                boundary_edges_ = np.vstack((boundary_edges_, [i, k]))
        boundary_triangles_ = np.vstack((boundary_triangles_, [-1, -1, -1]))
    else:
        # 3D tetrahedron mesh
        triangles = set()
        for [p0, p1, p2, p3] in mesh_elements:
            triangles.add((p0, p2, p1))
            triangles.add((p0, p3, p2))
            triangles.add((p0, p1, p3))
            triangles.add((p1, p2, p3))
        for (p0, p1, p2) in triangles:
            if (p0, p2, p1) not in triangles:
                if (p2, p1, p0) not in triangles:
                    if (p1, p0, p2) not in triangles:
                        boundary_points_.update([p0, p1, p2])
                        if p0 < p1:
                            boundary_edges_ = np.vstack((boundary_edges_, [p0, p1]))
                        if p1 < p2:
                            boundary_edges_ = np.vstack((boundary_edges_, [p1, p2]))
                        if p2 < p0:
                            boundary_edges_ = np.vstack((boundary_edges_, [p2, p0]))
                        boundary_triangles_ = np.vstack((boundary_triangles_, [p0, p1, p2]))
    return boundary_points_, boundary_edges_, boundary_triangles_
