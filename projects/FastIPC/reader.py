import meshio
import math
import numpy as np
from scipy.spatial.transform import Rotation


def read_msh(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    raw_particles = lines[lines.index('$Nodes\n') + 3:lines.index('$EndNodes\n')]
    raw_elements = lines[lines.index('$Elements\n') + 3:lines.index('$EndElements\n')]
    mesh_particles = np.array(list(map(lambda x: list(map(float, x[:-1].split(' ')[1:])), raw_particles)))
    mesh_elements = np.array(list(map(lambda x: list(map(int, x[:-1].split(' ')[1:])), raw_elements))) - 1
    return mesh_particles, mesh_elements


settings = {}


def init(dim):
    settings['dim'] = dim
    settings['mesh_particles'] = np.zeros((0, dim), dtype=np.float64)
    settings['mesh_elements'] = np.zeros((0, dim + 1), dtype=np.int32)
    settings['mesh_scale'] = 1.
    settings['mesh_offset'] = [0., 0., 0.]


def add_object(filename, translation=None, rotation=None, scale=None):
    translation = np.zeros(settings['dim']) if translation is None else np.array(translation)
    rotation = np.zeros(settings['dim']) if rotation is None else np.array(rotation)
    scale = np.ones(settings['dim']) if scale is None else np.array(scale)
    if filename[-4:] == '.msh':
        new_particles, new_elements = read_msh(filename)
    else:
        mesh = meshio.read(filename)
        new_particles = mesh.points
        new_elements = mesh.cells[0].data
    if settings['dim'] == 2:
        rotation *= np.pi / 180.
        rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                    [np.sin(rotation), np.cos(rotation)]])
    else:
        rotation *= np.pi / 180.
        rotation_matrix = Rotation.from_rotvec(rotation).as_matrix()
    n_particles = len(new_particles)
    for i in range(n_particles):
        p = new_particles[i, :]
        new_particles[i, :] = (p * scale) @ rotation_matrix + translation
    old_particles = settings['mesh_particles']
    old_elements = settings['mesh_elements']
    settings['mesh_particles'] = np.vstack((old_particles, new_particles))
    settings['mesh_elements'] = np.vstack((old_elements, new_elements + len(old_particles)))


def set_size(absolute_scale):
    mesh_particles = settings['mesh_particles']
    lower = np.amin(mesh_particles, axis=0)
    upper = np.amax(mesh_particles, axis=1)
    relative_scale = (upper - lower).max()
    settings['mesh_particles'] = mesh_particles / relative_scale * absolute_scale


def read(testcase):
    ##################################################### 3D #####################################################
    if testcase == 0:
        # two tets
        init(3)
        add_object('input/tet.vtk')
        add_object('input/tet.vtk', [0.501, -0.5, 0.5])
        settings['gravity'] = 0.
        settings['dirichlet_generator'] = lambda t: (np.zeros(len(settings['mesh_particles']), dtype=bool), settings['mesh_particles'])
        return settings
    if testcase == 1001:
        # two spheres
        init(3)
        add_object('input/sphere1K.vtk', [-0.51, 0, 0])
        add_object('input/sphere1K.vtk', [0.51, 0, 0])
        settings['gravity'] = 0.
        settings['dirichlet_generator'] = lambda t: (np.zeros(len(settings['mesh_particles']), dtype=bool), settings['mesh_particles'])
        return settings
    elif testcase == 1002:
        # mat twist
        init(3)
        # add_object('input/mat150x150t40.msh')
        add_object('input/mat20x20.vtk', scale=[1., 4., 1.])
        def dirichlet_generator(t):
            speed = math.pi * 0.4
            dirichlet_fixed = np.zeros(len(settings['mesh_particles']), dtype=bool)
            dirichlet_value = settings['mesh_particles'].copy()
            n_particles = len(dirichlet_value)
            for i in range(n_particles):
                if dirichlet_value[i][0] < -0.45 or dirichlet_value[i][0] > 0.45:
                    dirichlet_fixed[i] = True
                    a, b, c = dirichlet_value[i, 0], dirichlet_value[i, 1], dirichlet_value[i, 2]
                    angle = math.atan2(b, c)
                    angle += speed * t * (1 if a < 0 else -1)
                    radius = math.sqrt(b * b + c * c)
                    dirichlet_value[i, 0] = a
                    dirichlet_value[i, 1] = radius * math.sin(angle)
                    dirichlet_value[i, 2] = radius * math.cos(angle)
            return dirichlet_fixed, dirichlet_value
        settings['gravity'] = 0.
        settings['dirichlet_generator'] = dirichlet_generator
        return settings
    elif testcase == 1003:
        # sphere on mat
        init(3)
        add_object('input/sphere1K.msh', translation=[0., 1., 0.])
        add_object('input/mat40x40.msh', scale=[4., 4., 4.])
        def dirichlet_generator(t):
            dirichlet_fixed = np.zeros(len(settings['mesh_particles']), dtype=bool)
            dirichlet_value = settings['mesh_particles'].copy()
            n_particles = len(dirichlet_value)
            for i in range(n_particles):
                if dirichlet_value[i, 0] < -1.85 or dirichlet_value[i, 0] > 1.85:
                    dirichlet_fixed[i] = True
            return dirichlet_fixed, dirichlet_value
        settings['gravity'] = -9.8
        settings['dirichlet_generator'] = dirichlet_generator
        return settings
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
        mesh_particles = np.vstack((mesh.points * 200 + [-100, -220, -100], mesh.points * 10 + [-41.0, -11.4, -5.0],
                                    mesh.points * 10 + [31, -11.4, -5.0]))
        mesh_elements = np.vstack(
            (mesh.cells[0].data, mesh.cells[0].data + len(mesh.points), mesh.cells[0].data + len(mesh.points) * 2))
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
        mesh_particles = np.vstack((mesh.points * 40 + [-20, -40.44, -20], mesh.points * 0.4 + [0.2, 6, -0.2],
                                    mesh.points * 0.4 + [0.5, 9, -0.3]))
        mesh_elements = np.vstack(
            (mesh.cells[0].data, mesh.cells[0].data + len(mesh.points), mesh.cells[0].data + len(mesh.points) * 2))
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
