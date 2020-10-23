import meshio
import math
import numpy as np


def read_msh(fn):
    f = open(fn, 'r')
    lines = f.readlines()
    raw_particles = lines[lines.index('$Nodes\n') + 3:lines.index('$EndNodes\n')]
    raw_elements = lines[lines.index('$Elements\n') + 3:lines.index('$EndElements\n')]
    mesh_particles = np.array(list(map(lambda x: list(map(float, x[:-1].split(' ')[1:])), raw_particles)))
    mesh_elements = np.array(list(map(lambda x: list(map(int, x[:-1].split(' ')[1:])), raw_elements))) - 1
    return mesh_particles, mesh_elements


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
    elif testcase == 1002:
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
    elif testcase == 1003:
        # sphere on mat
        mesh_points0, mesh_elements0 = read_msh("input/sphere1K.msh")
        mesh_points1, mesh_elements1 = read_msh("input/mat40x40.msh")
        mesh_particles = np.vstack((mesh_points0 + [0, 1, 0], mesh_points1 * 4))
        offset = len(mesh_points0)
        mesh_elements = np.vstack((mesh_elements0, mesh_elements1 + offset))
        mesh_scale = 0.3
        mesh_offset = [0, -0.3, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        for i in range(n_particles):
            if mesh_particles[i][0] < -1.85 or mesh_particles[i][0] > 1.85:
                dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 3
    elif testcase == 1004:
        # mat on knife
        mesh_points, mesh_elements = read_msh("input/mat40x40.msh")
        mesh_particles = mesh_points * 1.2 + [0.5, 1, 0.1]
        offset = len(mesh_points)
        print('!!! offset ', offset)
        dxs = [0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9]
        orients = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for dx, ori in zip(dxs, orients):
            dy = -0.1
            dz = -0.7
            x = np.array([0.0, 0.0, 0.0]) + np.array([dx, dy, dz])
            y = np.array([0.0, 0.0, 1.0]) + np.array([dx, dy, dz])
            z = np.array([0.0, 1.0, ori]) + np.array([dx, dy, dz])
            mesh_particles = np.vstack((mesh_particles, [x, y, z]))
        mesh_scale = 0.6
        mesh_offset = [0, -0.3, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        for i in range(offset, n_particles):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 3
    elif testcase == 1005:
        # sphere on points
        mesh_points, mesh_elements = read_msh("input/sphere5K.msh")
        mesh_particles = mesh_points * 3 + [2, 2, 2]
        offset = len(mesh_points)
        print('!!! offset ', offset)
        for i in range(20):
            for j in range(20):
                x, y, z = 0.2 * i, 0.0, 0.2 * j
                mesh_particles = np.vstack((mesh_particles, [x, y, z]))
        mesh_scale = 0.6
        mesh_offset = [0, -0.3, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        for i in range(offset, n_particles):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 3
    elif testcase == 1006:
        # arch
        mesh = meshio.read("input/cube.vtk")
        mesh_particles = np.vstack((mesh.points * 200 + [-100, -220, -100], mesh.points * 10 + [-41.0, -11.4, -5.0], mesh.points * 10 + [31, -11.4, -5.0]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh.points), mesh.cells[0].data + len(mesh.points) * 2))
        for i in range(25):
            tmp_particles, tmp_elements = read_msh("input/arch/largeArch." + str(i + 1).zfill(2) + ".msh")
            dx = -1.2 + 0.1 * i
            dy = -1.2 + 0.1 * i
            if i >= 12:
                dy = -dy
            offset = len(mesh_particles)
            mesh_particles = np.vstack((mesh_particles, tmp_particles + [dx, dy, 0.0]))
            mesh_elements = np.vstack((mesh_elements, tmp_elements + offset))
        mesh_scale = 0.6
        mesh_offset = [0, -0.3, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        for i in range(24):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 3
    elif testcase == 1007:
        # card house
        def rotate(x, y, alpha):
            xx = math.cos(alpha) * x - math.sin(alpha) * y
            yy = math.sin(alpha) * x + math.cos(alpha) * y
            return xx, yy
        mesh = meshio.read("input/cube.vtk")
        mesh_particles = np.vstack((mesh.points * 40 + [-20, -40.44, -20], mesh.points * 0.4 + [0.2, 6, -0.2], mesh.points * 0.4 + [0.5, 9, -0.3]))
        mesh_elements = np.vstack((mesh.cells[0].data, mesh.cells[0].data + len(mesh.points), mesh.cells[0].data + len(mesh.points) * 2))
        mesh = meshio.read("input/mat20x20.vtk")
        tmp_particles = mesh.points
        tmp_elements = mesh.cells[0].data
        angles = [math.pi / 3, -math.pi / 3, math.pi / 3, -math.pi / 3, 0.0, math.pi / 3, -math.pi / 3]
        dxs = [0.0, 0.515, 1.03, 1.545, 0.78, 0.515, 1.03]
        dys = [0.0, 0.0, 0.0, 0.0, 0.445, 0.89, 0.89]
        for dx, dy, a in zip(dxs, dys, angles):
            offset = len(mesh_particles)
            tmp = tmp_particles.copy()
            if a == 0.0:
                tmp *= [1.1, 1, 1.1]
            for i in range(len(tmp)):
                x, y, z = tmp[i, 0], tmp[i, 1], tmp[i, 2]
                xx, yy = rotate(x, y, a)
                tmp[i, 0], tmp[i, 1], tmp[i, 2] = xx, yy, z
            mesh_particles = np.vstack((mesh_particles, tmp + [dx, dy, 0.0]))
            mesh_elements = np.vstack((mesh_elements, tmp_elements + offset))
        mesh_scale = 0.6
        mesh_offset = [0, -0.3, 0]
        n_particles = len(mesh_particles)
        dirichlet_fixed = np.zeros(n_particles, dtype=bool)
        dirichlet_value = mesh_particles
        for i in range(8):
            dirichlet_fixed[i] = True
        return mesh_particles, mesh_elements, mesh_scale, mesh_offset, dirichlet_fixed, dirichlet_value, -9.8, 3

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
