import meshio
import numpy as np


def read(testcase):
    ##################################################### 3D #####################################################
    if testcase == 0:
        # two spheres
        mesh = meshio.read("input/sphere1K.vtk")
        mesh_particles = np.vstack((mesh.points + [-0.51, 0, 0], mesh.points + [0.51, 0, 0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.8
        mesh_offset = [0, 0, 0]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 3, dtype=bool)
        dirichlet_value = np.zeros(len(mesh_particles) * 3, dtype=np.float32)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 3
    elif testcase == 1:
        mesh = meshio.read("input/mat20x20.vtk")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.8
        mesh_offset = [0, 0, 0]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 3, dtype=bool)
        for i in range(len(mesh_particles)):
            if mesh_particles[i][0] < -0.45 or mesh_particles[i][0] > 0.45:
                dirichlet_fixed[i * 3] = True
                dirichlet_fixed[i * 3 + 1] = True
                dirichlet_fixed[i * 3 + 2] = True
                print(i, mesh_particles[i][0], mesh_particles[i][1], mesh_particles[i][2])
        dirichlet_value = np.zeros(len(mesh_particles) * 3, dtype=np.float32)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 3
    elif testcase == 2:
        # one spheres
        mesh = meshio.read("input/sphere1K.vtk")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.8
        mesh_offset = [0, 0, 0]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 3, dtype=bool)
        dirichlet_value = np.zeros(len(mesh_particles) * 3, dtype=np.float32)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 3
    elif testcase == 3:
        # two tets
        mesh = meshio.read("input/tet.vtk")
        mesh_particles = np.vstack((mesh.points + [0, 0, 0], mesh.points + [0.501, -0.5, 0.5]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.6
        mesh_offset = [0, 0, 0]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 3, dtype=bool)
        dirichlet_value = np.zeros(len(mesh_particles) * 3, dtype=np.float32)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 3
    ##################################################### 2D #####################################################
    elif testcase == 4:
        # two triangles
        mesh = meshio.read("input/cubes.obj")
        mesh_particles = np.vstack((mesh.points + [0, 0, 0], mesh.points + [1, -0.5, 0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.2
        mesh_offset = [0.4, 0.5]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        dirichlet_value = np.zeros(len(mesh_particles) * 2, dtype=np.float32)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
    elif testcase == 5:
        # two triangles
        mesh = meshio.read("input/sphere.obj")
        mesh_particles = np.vstack((mesh.points + [0, 0, 0], mesh.points + [0.21, 0, 0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.8
        mesh_offset = [0.4, 0.5]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        dirichlet_value = np.zeros(len(mesh_particles) * 2, dtype=np.float32)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
