import meshio
import numpy as np


def read(testcase):
    ##################################################### 3D #####################################################
    if testcase == 0:
        # two tets
        mesh = meshio.read("input/tet.vtk")
        mesh_particles = np.vstack((mesh.points + [0, 0, 0], mesh.points + [0.501, -0.5, 0.5]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.6
        mesh_offset = [0, 0, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 0.0, 3
    if testcase == 1:
        # two spheres
        mesh = meshio.read("input/sphere1K.vtk")
        mesh_particles = np.vstack((mesh.points + [-0.51, 0, 0], mesh.points + [0.51, 0, 0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.8
        mesh_offset = [0, 0, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 0.0, 3
    elif testcase == 2:
        # mat twist
        mesh = meshio.read("input/mat20x20.vtk")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.8
        mesh_offset = [0, 0, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        for i in range(n_particles):
            if mesh_particles[i][0] < -0.45 or mesh_particles[i][0] > 0.45:
                dirichlet_fixed[i] = True
                print(i, mesh_particles[i][0], mesh_particles[i][1], mesh_particles[i][2])
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 0.0, 3
    elif testcase == 3:
        # sphere on mat
        mesh0 = meshio.read("input/sphere1K.msh")
        print(mesh0.points)
        print(mesh0.cells)
        mesh1 = meshio.read("input/mat40x40.msh")
        mesh_particles = np.vstack((mesh0.points, mesh1.points + [0, 1, 0]))
        offset = len(mesh0.points)
        mesh_elements = np.vstack((mesh0.cells[0].data, mesh1.cells[0].data + offset))
        mesh_scale = 0.5
        mesh_offset = [0, 0, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        # for i in [954, 955, 956, 957, 958]:
        #     dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2
    ##################################################### 2D #####################################################
    elif testcase == 4:
        # two triangles
        mesh = meshio.read("input/cubes.obj")
        mesh_particles = np.vstack((mesh.points + [0, 0, 0], mesh.points + [1, -0.5, 0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.2
        mesh_offset = [0.4, 0.5]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 0.0, 2
    elif testcase == 5:
        # two spheres
        mesh = meshio.read("input/sphere.obj")
        mesh_particles = np.vstack((mesh.points + [0, 0, 0], mesh.points + [0.5, 0, 0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh_particles) / 2))
        mesh_scale = 0.8
        mesh_offset = [0.4, 0.5]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in range(2):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 0.0, 2
    elif testcase == 6:
        # two spheres
        mesh = meshio.read("input/Sharkey.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.35, 0.3]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in range(12):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2
    elif testcase == 7:
        # two spheres
        mesh = meshio.read("input/Sharkey_floor.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.32, 0.3]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in [954, 955, 956, 957]:
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2
    elif testcase == 8:
        # two spheres
        mesh = meshio.read("input/Sharkey_valley.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.32, 0.3]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in [954, 955, 956, 957, 958]:
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2
    elif testcase == 9:
        # two spheres
        mesh0 = meshio.read("input/Sharkey_valley.obj")
        mesh1 = meshio.read("input/Sharkey.obj")
        mesh_particles = np.vstack((mesh0.points, mesh1.points + [0, 1, 0]))
        offset = len(mesh0.points)
        mesh_elements = np.vstack((mesh0.cells[0].data, mesh1.cells[0].data + offset))
        mesh_scale = 0.4
        mesh_offset = [0.38, 0.2]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in [954, 955, 956, 957, 958]:
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2
    elif testcase == 10:
        # two spheres
        mesh = meshio.read("input/Sharkey.obj")
        mesh_particles = np.vstack((mesh.points, [
            [1.1, 0.65, 0], [2.1, 0.65, 0], [0.5, 0.95, 0], [2.7, 0.95, 0],
            [0.5, 0.25, 0], [2.7, 0.25, 0], [1.1, 0.55, 0], [2.1, 0.55, 0]
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
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        print("!!!! offset : ", offset)
        for i in range(len(mesh_particles)):
            if mesh_particles[i, 0] > 0.745 and i < offset:
                dirichlet_fixed[i] = True
        for i in range(offset, offset + 8):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, 0.0, 2
    elif testcase == 11:
        mesh = meshio.read("input/noodles.obj")
        mesh_particles = mesh.points
        mesh_elements = mesh.cells[0].data
        mesh_scale = 0.6
        mesh_offset = [0.32, 0.3]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in [4000, 4001, 4004, 4005]:
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2
    elif testcase == 12:
        mesh = meshio.read("input/fluffy.obj")
        mesh_particles = np.vstack((mesh.points, [
            [-1, -0.8, 0], [1, -0.8, 0], [1, -0.7, 0], [-1, -0.7, 0]
        ]))
        offset = len(mesh.points)
        mesh_elements = np.vstack((mesh.cells[0].data, [
            [offset, offset + 1, offset + 2],
            [offset + 0, offset + 2, offset + 3]
        ]))
        mesh_scale = 0.5
        mesh_offset = [0.5, 0.6]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in range(offset, offset + 4):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2
    elif testcase == 13:
        # two spheres
        mesh = meshio.read("input/items.obj")
        mesh_particles = np.vstack((mesh.points, [
            [-0.5, -0.3, 0],
            [0.3, -0.3, 0],
            [-0.5, 0.6, 0],
            [1.1, -0.3, 0],
            [1.1, 0.6, 0]
        ]))
        offset = len(mesh.points)
        mesh_elements = np.vstack((mesh.cells[0].data, [
            [offset, offset + 1, offset + 2],
            [offset + 1, offset + 3, offset + 4]
        ]))
        mesh_scale = 0.4
        mesh_offset = [0.38, 0.2]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles[:, :2]
        for i in range(offset, offset + 5):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 2