import meshio
import math
import numpy as np
from common.math.graph_tools import *
from scipy.spatial.transform import Rotation
import os, sys


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
        if settings['dim'] == 3:
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
        else:
            print("Not implemented!!")

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


def read():
    ################################################### GENERAL ##################################################
    testcase = int(sys.argv[1])
    # settings['start_frame'] = int(sys.argv[2])
    # settings['dt'] = float(sys.argv[3])
    # settings['E'] = float(sys.argv[4])
    # settings['scale'] = float(sys.argv[5])
    settings['dt'] = 0.04
    settings['E'] = 1.e4
    settings['scale'] = .5

    directory = 'output/' + '_'.join(sys.argv[:2] + sys.argv[3:]) + '/'
    os.makedirs(directory + 'images/', exist_ok=True)
    os.makedirs(directory + 'caches/', exist_ok=True)
    os.makedirs(directory + 'objs/', exist_ok=True)
    print('output directory:', directory)
    settings['directory'] = directory

    ##################################################### 3D #####################################################
    

    ##################################################### 2D #####################################################
    elif testcase == 1:
        # hang sharkey
        init(2)
        settings['gravity'] = -9.8
        add_object('input/Sharkey.obj')
        settings['dirichlet'] = lambda t: (np.concatenate(([True] * 12, [False] * (len(settings['mesh_particles']) - 12))), settings['mesh_particles'])
        adjust_camera()
        settings['mesh_scale'] *= 0.8
        settings['mesh_offset'] += [0.1, 0.25]
        return settings
    elif testcase == 2:
        # one sharkey
        init(2)
        settings['gravity'] = -9.8
        add_object('input/Sharkey.obj')
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
        add_boundary([[-0.5, 0.6], [0.3, -0.3], [1.1, 0.6]])
        settings['dirichlet'] = lambda t: (np.concatenate(([False] * (len(settings['mesh_particles']) - 3), [True] * 3)), settings['mesh_particles'])
        adjust_camera()
        return settings
    elif testcase == 4:
        # pull sharkey
        init(2)
        settings['gravity'] = 0
        add_object('input/Sharkey.obj')

        thickness = 0.2
        add_boundary([[0.5, 0.95], [1.1, 0.6 + thickness * 0.5], [2.1, 0.6 + thickness * 0.5], [2.7, 0.95]])
        add_boundary([[0.5, 0.25], [1.1, 0.6 - thickness * 0.5], [2.1, 0.6 - thickness * 0.5], [2.7, 0.25]])

        def dirichlet_generator(t):
            dirichlet_fixed = np.zeros(len(settings['mesh_particles']), dtype=bool)
            dirichlet_value = settings['mesh_particles'].copy()
            dirichlet_fixed[-8:] = True
            n_particles = len(dirichlet_value)
            for i in range(n_particles - 8):
                if dirichlet_value[i][0] > 0.745:
                    dirichlet_fixed[i] = True
                    dirichlet_value[i, 0] += t
            return dirichlet_fixed, dirichlet_value
        settings['dirichlet'] = dirichlet_generator
        adjust_camera()
        settings['mesh_scale'] *= 1.5
        settings['mesh_offset'] += [0., 0.]
        return settings


