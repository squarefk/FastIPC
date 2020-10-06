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
        # two spheres
        mesh = meshio.read("input/sphere.obj")
        mesh_particles = np.vstack((mesh.points + [0, 0, 0], mesh.points + [0.5, 0, 0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.8
        mesh_offset = [0.4, 0.5]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        for i in range(2 * 2):
            dirichlet_fixed[i] = True
        dirichlet_value = mesh_particles[:, :2].reshape((len(mesh_particles) * 2))
        print(mesh_particles)
        print(dirichlet_value)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
    elif testcase == 6:
        # two spheres
        mesh = meshio.read("input/Sharkey.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.35, 0.3]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        for i in range(12 * 2):
            dirichlet_fixed[i] = True
        dirichlet_value = mesh_particles[:, :2].reshape((len(mesh_particles) * 2))
        print(mesh_particles)
        print(dirichlet_value)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
    elif testcase == 7:
        # two spheres
        mesh = meshio.read("input/Sharkey_floor.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.32, 0.3]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        for i in [954, 955]:
            dirichlet_fixed[i * 2] = True
            dirichlet_fixed[i * 2 + 1] = True
        dirichlet_value = mesh_particles[:, :2].reshape((len(mesh_particles) * 2))
        print(mesh_particles)
        print(dirichlet_value)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
    elif testcase == 8:
        # two spheres
        mesh = meshio.read("input/Sharkey_valley.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.32, 0.3]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        for i in [954, 957]:
            dirichlet_fixed[i * 2] = True
            dirichlet_fixed[i * 2 + 1] = True
        dirichlet_value = mesh_particles[:, :2].reshape((len(mesh_particles) * 2))
        print(mesh_particles)
        print(dirichlet_value)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
    elif testcase == 9:
        # two spheres
        mesh0 = meshio.read("input/Sharkey_valley.obj")
        mesh1 = meshio.read("input/Sharkey.obj")
        mesh_particles = np.vstack((mesh0.points, mesh1.points + [0, 1, 0]))
        offset = len(mesh0.points)
        mesh_elements = np.vstack((mesh0.cells[0].data, mesh1.cells[0].data + offset))
        mesh_scale = 0.4
        mesh_offset = [0.38, 0.2]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        for i in [954, 957]:
            dirichlet_fixed[i * 2] = True
            dirichlet_fixed[i * 2 + 1] = True
        dirichlet_value = mesh_particles[:, :2].reshape((len(mesh_particles) * 2))
        print(mesh_particles)
        print(dirichlet_value)
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
    elif testcase == 10:
        # two spheres
        mesh = meshio.read("input/Sharkey.obj")
        mesh_particles = np.vstack((mesh.points, [
            [1.1, 0.65, 0], [2.1, 0.65, 0], [0.8, 0.75, 0], [2.4, 0.75, 0],
            [0.8, 0.45, 0], [2.4, 0.45, 0], [1.1, 0.55, 0], [2.1, 0.55, 0]
        ]))
        offset = len(mesh.points)
        mesh_elements = np.vstack((mesh.cells[0].data, [
            [offset, offset + 1, offset + 2],
            [offset + 2, offset + 1, offset + 3],
            [offset + 4, offset + 5, offset + 6],
            [offset + 6, offset + 5, offset + 7]
        ]))
        mesh_scale = 0.3
        mesh_offset = [0.02, 0.3]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        print("!!!! offset : ", offset)
        for i in range(len(mesh_particles)):
            if mesh_particles[i, 0] > 0.745 and i < offset:
                dirichlet_fixed[i * 2] = True
                dirichlet_fixed[i * 2 + 1] = True
        for i in [offset + 2, offset + 3, offset + 4, offset + 5]:
            dirichlet_fixed[i * 2] = True
            dirichlet_fixed[i * 2 + 1] = True
        dirichlet_value = mesh_particles[:, :2].reshape((len(mesh_particles) * 2))
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2
    elif testcase == 11:
        mesh = meshio.read("input/noodles.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.32, 0.3]
        dirichlet_fixed = np.zeros(len(mesh_particles) * 2, dtype=bool)
        for i in [4000, 4001, 4004, 4005]:
            dirichlet_fixed[i * 2] = True
            dirichlet_fixed[i * 2 + 1] = True
        dirichlet_value = mesh_particles[:, :2].reshape((len(mesh_particles) * 2))
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 2