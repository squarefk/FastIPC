import meshio
import math
import numpy as np
from common.math.graph_tools import *
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
    settings['mesh_offset'] = [0., 0.] if dim == 2 else [0., 0., 0.]


def add_object(filename, translation=None, rotation=None, scale=None):
    translation = np.zeros(settings['dim']) if translation is None else np.array(translation)
    if settings['dim'] == 2:
        rotation = 0. if rotation is None else rotation
    else:
        rotation = np.zeros(settings['dim'] * 2 - 3) if rotation is None else np.array(rotation)
    scale = np.ones(settings['dim']) if scale is None else np.array(scale)
    if filename == 'cube':
        n = 10
        new_particles = np.zeros((n**3, 3), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    new_particles[i * n * n + j * n + k, 0] = i * 0.1
                    new_particles[i * n * n + j * n + k, 1] = j * 0.1
                    new_particles[i * n * n + j * n + k, 2] = k * 0.1
        new_elements = np.zeros(((n - 1)**3 * 6, 4), dtype=np.int32)
        for i in range(n - 1):
            for j in range(n - 1):
                for k in range(n - 1):
                    f = np.array([(i + 0) * n * n + (j + 0) * n + k + 0,
                                  (i + 1) * n * n + (j + 0) * n + k + 0,
                                  (i + 1) * n * n + (j + 0) * n + k + 1,
                                  (i + 0) * n * n + (j + 0) * n + k + 1,
                                  (i + 0) * n * n + (j + 1) * n + k + 0,
                                  (i + 1) * n * n + (j + 1) * n + k + 0,
                                  (i + 1) * n * n + (j + 1) * n + k + 1,
                                  (i + 0) * n * n + (j + 1) * n + k + 1])
                    new_elements[i * (n - 1) * (n - 1) * 6 + j * (n - 1) * 6 + k * 6 + 0, :] = np.array([f[0], f[4], f[6], f[5]], dtype=np.int32)
                    new_elements[i * (n - 1) * (n - 1) * 6 + j * (n - 1) * 6 + k * 6 + 1, :] = np.array([f[3], f[6], f[2], f[0]], dtype=np.int32)
                    new_elements[i * (n - 1) * (n - 1) * 6 + j * (n - 1) * 6 + k * 6 + 2, :] = np.array([f[0], f[4], f[7], f[6]], dtype=np.int32)
                    new_elements[i * (n - 1) * (n - 1) * 6 + j * (n - 1) * 6 + k * 6 + 3, :] = np.array([f[3], f[6], f[0], f[7]], dtype=np.int32)
                    new_elements[i * (n - 1) * (n - 1) * 6 + j * (n - 1) * 6 + k * 6 + 4, :] = np.array([f[2], f[0], f[6], f[1]], dtype=np.int32)
                    new_elements[i * (n - 1) * (n - 1) * 6 + j * (n - 1) * 6 + k * 6 + 5, :] = np.array([f[6], f[0], f[5], f[1]], dtype=np.int32)
    elif filename[-4:] == '.msh':
        new_particles, new_elements = read_msh(filename)
    else:
        mesh = meshio.read(filename)
        new_particles = mesh.points[:, :settings['dim']]
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
        new_particles[i, :] = rotation_matrix @ (p * scale) + translation
    old_particles = settings['mesh_particles']
    old_elements = settings['mesh_elements']
    settings['mesh_particles'] = np.vstack((old_particles, new_particles))
    settings['mesh_elements'] = np.vstack((old_elements, new_elements + len(old_particles)))


def add_boundary(positions):
    old_particles = settings['mesh_particles']
    settings['mesh_particles'] = np.vstack((old_particles, positions))
    offset = len(old_particles)
    if 'boundary' not in settings:
        print('Please do find_boundary first')
    boundary_points, boundary_edges, boundary_triangles = settings['boundary']
    if len(positions) == 1:
        boundary_points.update([offset])
    elif len(positions) == 2:
        boundary_points.update([offset, offset + 1])
        boundary_edges = np.vstack((boundary_edges, [offset, offset + 1]))
    else:
        boundary_points.update([offset, offset + 1, offset + 2])
        boundary_edges = np.vstack((boundary_edges, [offset, offset + 1]))
        boundary_edges = np.vstack((boundary_edges, [offset + 1, offset + 2]))
        boundary_edges = np.vstack((boundary_edges, [offset + 2, offset]))
        boundary_triangles = np.vstack((boundary_triangles, [offset, offset + 1, offset + 2]))
    settings['boundary'] = (boundary_points, boundary_edges, boundary_triangles)


def set_size(absolute_scale):
    mesh_particles = settings['mesh_particles']
    lower = np.amin(mesh_particles, axis=0)
    upper = np.amax(mesh_particles, axis=0)
    relative_scale = (upper - lower).max()
    settings['mesh_particles'] = mesh_particles / relative_scale * absolute_scale


def adjust_camera():
    mesh_particles = settings['mesh_particles']
    dim = settings['dim']
    lower = np.amin(mesh_particles, axis=0)
    upper = np.amax(mesh_particles, axis=0)
    if dim == 2:
        settings['mesh_scale'] = 0.8 / (upper - lower).max()
        settings['mesh_offset'] = [0.5, 0.5] - ((upper + lower) * 0.5) * settings['mesh_scale']
    else:
        settings['mesh_scale'] = 1.6 / (upper - lower).max()
        settings['mesh_offset'] = - ((upper + lower) * 0.5) * settings['mesh_scale']


def read(testcase):
    ##################################################### 3D #####################################################
    if testcase == 1001:
        # two spheres
        init(3)
        settings['gravity'] = 0.
        add_object('input/sphere1K.vtk', [-0.51, 0, 0])
        add_object('input/sphere1K.vtk', [0.51, 0, 0])
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        settings['dirichlet'] = lambda t: (np.zeros(len(settings['mesh_particles']), dtype=bool), settings['mesh_particles'])
        return settings
    elif testcase == 1002:
        # mat twist
        init(3)
        settings['gravity'] = 0.
        # add_object('input/mat150x150t40.msh')
        add_object('input/mat20x20.vtk', scale=[1., 4., 1.])
        settings['boundary'] = find_boundary(settings['mesh_elements'])

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
        settings['dirichlet'] = dirichlet_generator
        return settings
    elif testcase == 1003:
        # sphere on mat
        init(3)
        settings['gravity'] = -9.8
        add_object('input/sphere1K.msh', translation=[0., 1., 0.])
        add_object('input/mat40x40.msh', scale=[4., 4., 4.])
        settings['boundary'] = find_boundary(settings['mesh_elements'])

        def dirichlet_generator(t):
            dirichlet_fixed = np.zeros(len(settings['mesh_particles']), dtype=bool)
            dirichlet_value = settings['mesh_particles'].copy()
            n_particles = len(dirichlet_value)
            for i in range(n_particles):
                if dirichlet_value[i, 0] < -1.85 or dirichlet_value[i, 0] > 1.85:
                    dirichlet_fixed[i] = True
            return dirichlet_fixed, dirichlet_value
        settings['dirichlet'] = dirichlet_generator
        return settings
    elif testcase == 1004:
        # mat on knife
        init(3)
        settings['gravity'] = -9.8
        add_object('input/mat40x40.msh', translation=[0.5, 1, 0.1], scale=[1.2, 1.2, 1.2])

        settings['boundary'] = find_boundary(settings['mesh_elements'])
        dxs = [0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9]
        orients = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for dx, ori in zip(dxs, orients):
            dy = -0.1
            dz = -0.7
            x = np.array([0.0, 0.0, 0.0]) + np.array([dx, dy, dz])
            y = np.array([0.0, 0.0, 1.0]) + np.array([dx, dy, dz])
            z = np.array([0.0, 1.0, ori]) + np.array([dx, dy, dz])
            add_boundary([x, y, z])

        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 27), [True] * 27)), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 1005:
        # sphere on points
        init(3)
        settings['gravity'] = -9.8
        add_object('input/sphere5K.msh', translation=[2., 2., 2.], scale=[3., 3., 3.])

        settings['boundary'] = find_boundary(settings['mesh_elements'])
        for i in range(20):
            for j in range(20):
                x, y, z = 0.2 * i, 0.0, 0.2 * j
                add_boundary([[x, y, z]])

        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 400), [True] * 400)), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 1006:
        # arch
        init(3)
        settings['gravity'] = -9.8
        for i in range(25):
            dx = -1.2 + 0.1 * i
            dy = -1.2 + 0.1 * i
            if i >= 12:
                dy = -dy
            add_object('input/arch/largeArch.' + str(i + 1).zfill(2) + '.msh', translation=[dx, dy, 0.])
        add_object('input/cube.vtk', translation=[-100., -21., -100.], scale=[200., 1., 200.])
        add_object('input/cube.vtk', translation=[-41.0, -11.4, -5.0], scale=[10., 10., 10.])
        add_object('input/cube.vtk', translation=[31, -11.4, -5.0], scale=[10., 10., 10.])
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 24), [True] * 24)), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 1007:
        # card house
        init(3)
        settings['gravity'] = -9.8

        angles = [60., -60, 60., -60., 0., 60., -60.]
        dxs = [0.0, 0.515, 1.03, 1.545, 0.78, 0.515, 1.03]
        dys = [0.0, 0.0, 0.0, 0.0, 0.445, 0.89, 0.89]
        for dx, dy, a in zip(dxs, dys, angles):
            scale = [1., 1., 1.]
            if a == 0.0:
                scale = [1.1, 1, 1.1]
            add_object('input/mat20x20.vtk', translation=[dx, dy, 0.], rotation=[0., 0., a], scale=scale)
        add_object('input/cube.vtk', translation=[0.2, 6, -0.2], scale=[0.4, 0.4, 0.4])
        add_object('input/cube.vtk', translation=[0.5, 9, -0.3], scale=[0.4, 0.4, 0.4])
        add_object('input/cube.vtk', translation=[-4, -1.44, -4], scale=[8., 1., 8.])
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 8), [True] * 8)), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 1008:
        # cube
        scale = 0.5
        init(3)
        settings['gravity'] = -9.8
        add_object('cube', translation=[-0.5 * scale, 0.5 * scale, -0.5 * scale], scale=[1. * scale, 1. * scale, 1. * scale])
        add_object('cube', translation=[0 * scale, 1.75 * scale, -0.5 * scale], scale=[1. * scale, 1. * scale, 1. * scale])
        add_object('cube', translation=[0.5 * scale, 3.0 * scale, -0.5 * scale], scale=[1. * scale, 1. * scale, 1. * scale])
        add_object('input/cube.vtk', translation=[-3. * scale, -0.1 * scale, -3. * scale], scale=[6. * scale, 0.1 * scale, 6. * scale])
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 8), [True] * 8)), settings['mesh_particles'])
        adjust_camera()
        return settings

    ##################################################### 2D #####################################################
    elif testcase == 1:
        # hang sharkey
        init(2)
        settings['gravity'] = -9.8
        add_object('input/Sharkey.obj')
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        settings['dirichlet'] = lambda t: (np.concatenate(([True] * 12, [False] * (len(settings['mesh_particles']) - 12))), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 2:
        # one sharkey
        init(2)
        settings['gravity'] = -9.8
        add_object('input/Sharkey.obj')
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        add_boundary([[-0.5, -0.1], [1.1, -0.1]])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 2), [True] * 2)), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 3:
        # two sharkey
        init(2)
        settings['gravity'] = -9.8
        add_object('input/Sharkey.obj')
        add_object('input/Sharkey.obj', translation=[0., 1.])
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        add_boundary([[-0.5, 0.6], [0.3, -0.3]])
        add_boundary([[0.3, -0.3], [1.1, 0.6]])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 4), [True] * 4)), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 4:
        # pull sharkey
        init(2)
        settings['gravity'] = 0
        add_object('input/Sharkey.obj')
        settings['boundary'] = find_boundary(settings['mesh_elements'])

        thickness = 0.1
        add_boundary([[0.5, 0.95], [1.1, 0.6 + thickness * 0.5]])
        add_boundary([[1.1, 0.6 + thickness * 0.5], [2.1, 0.6 + thickness * 0.5]])
        add_boundary([[2.1, 0.6 + thickness * 0.5], [2.7, 0.95]])
        add_boundary([[0.5, 0.25], [1.1, 0.6 - thickness * 0.5]])
        add_boundary([[1.1, 0.6 - thickness * 0.5], [2.1, 0.6 - thickness * 0.5]])
        add_boundary([[2.1, 0.6 - thickness * 0.5], [2.7, 0.25]])

        def dirichlet_generator(t):
            dirichlet_fixed = np.zeros(len(settings['mesh_particles']), dtype=bool)
            dirichlet_value = settings['mesh_particles'].copy()
            dirichlet_fixed[-12:] = True
            n_particles = len(dirichlet_value)
            for i in range(n_particles - 12):
                if dirichlet_value[i][0] > 0.745:
                    dirichlet_fixed[i] = True
                    dirichlet_value[i, 0] += t
            return dirichlet_fixed, dirichlet_value
        settings['dirichlet'] = dirichlet_generator
        adjust_camera()
        return settings
    elif testcase == 5:
        # noodles
        init(2)
        settings['gravity'] = -9.8
        add_object('input/noodles.obj')
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 8), [True] * 8)), settings['mesh_particles'])
        return settings
    elif testcase == 6:
        # fluffy
        init(2)
        settings['gravity'] = -9.8
        add_object('input/fluffy.obj')
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        add_boundary([[-1, -0.7], [1, -0.7]])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 2), [True] * 2)), settings['mesh_particles'])
        adjust_camera()
        return settings
