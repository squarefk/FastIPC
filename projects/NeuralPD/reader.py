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

    ##################################################### 3D #####################################################
    

    ##################################################### 2D #####################################################
    if testcase == 1:
        # stretch box
        init(2)
        settings['gravity'] = 0
        add_object('input/square.obj')
        def dirichlet(t):
            x = settings['mesh_particles']
            left_boundary = x[:, 0] < 0.00001
            right_boundary = x[:, 0] > 0.99999
            target_x = settings['mesh_particles'].copy()
            if t > 1:
                target_x[left_boundary, 0] -= 0.1 * (t-1)
                target_x[right_boundary, 0] += 0.1 * (t-1)
            fixed = np.logical_or(left_boundary, right_boundary)
            return fixed, target_x
        settings['dirichlet'] = dirichlet
        adjust_camera()
        settings['mesh_scale'] *= 0.1
        settings['mesh_offset'] += [0.35, 0.5]
        return settings

