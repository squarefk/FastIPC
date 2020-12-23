import meshio
import math
import numpy as np
from common.math.graph_tools import *
from scipy.spatial.transform import Rotation
import os, sys
import json

def read_msh(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    raw_particles = lines[lines.index('$Nodes\n') + 3:lines.index('$EndNodes\n')]
    raw_elements = lines[lines.index('$Elements\n') + 3:lines.index('$EndElements\n')]
    mesh_particles = np.array(list(map(lambda x: list(map(float, x[:-1].split(' ')[1:])), raw_particles)))
    mesh_elements = np.array(list(map(lambda x: list(map(int, x[:-1].split(' ')[1:])), raw_elements))) - 1
    return mesh_particles, mesh_elements


settings = {}

def extract_edges():
    mesh_elements = settings["mesh_elements"]
    edges = {}
    for [i, j, k] in mesh_elements:
        edges[(i,j)] = [k]
        edges[(j,k)] = [i]
        edges[(k,i)] = [j]
    for edge in edges:
        if len(edges[edge]) == 0:
            continue
        if edge[::-1] not in edge:
            continue
    # TODO

def parse_fold(filename):
    edge_types = {}
    with open(filename) as f:
        data = json.load(f)
    mesh_elements = np.array(data["faces_vertices"], dtype=np.int32)
    edges = {}
    for [i, j, k] in mesh_elements:
        edges[(i,j)] = [k]
        edges[(j,k)] = [i]
        edges[(k,i)] = [j]
    for edge in edges:
        if len(edges[edge]) == 0:
            continue
        if edge[::-1] not in edges:
            edges[edge].append(-1)
            continue
        edges[edge].append(edges[edge[::-1]].pop())

    for edge, t in zip(data["edges_vertices"], data["edges_assignment"]):
        edge_types[tuple(edge)] = -1 if t == "B" else 0 if t == "F" else 1 if t == "M" else 2
        edge_types[tuple(edge[::-1])] = edge_types[tuple(edge)]
    
    mesh_edges = []
    for edge in edges: 
        if len(edges[edge]) == 0:
            continue
        edge_info = list(edge) + edges[edge] + [edge_types[tuple(edge)]]
        mesh_edges.append(edge_info)
    settings["mesh_particles"] = np.array(data["vertices_coords"], dtype=np.float64)
    settings["mesh_elements"] = np.array(data["faces_vertices"], dtype=np.int32)
    settings["mesh_edges"] = np.array(mesh_edges, dtype=np.int32)
    

def init(dim):
    settings['dim'] = dim
    settings['mesh_particles'] = np.zeros((0, dim), dtype=np.float64)
    settings['mesh_elements'] = np.zeros((0, dim + 1), dtype=np.int32)
    settings['mesh_scale'] = 1.
    settings['mesh_offset'] = [0., 0.] if dim == 2 else [0., 0., 0.]
    settings['mesh_edges'] = np.zeros((0, 5), dtype=np.int32) 
        # [v0, v1, v2, v3, t] where (v0 v1) is the edge, v2 v3 are two vertices beside the edge. 
        # v0 v1 v2 are in the counter-clockwise order.
        # t is crease type: 0 - inner edges; 1 - M; 2 - V


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
            v = 1
            t1 = 0.5
            t2 = 1.0 + 2
            t3 = 1.5 + 2
            if t < t1:
                fixed = np.logical_or(left_boundary, right_boundary)
            elif t < t2:
                target_x[left_boundary, 0] -= v * (t-t1)
                target_x[right_boundary, 0] += v * (t-t1)
                fixed = np.logical_or(left_boundary, right_boundary)
            elif t < t3:
                target_x[left_boundary, 0] -= v * (t2-t1)
                target_x[right_boundary, 0] += v * (t2-t1)
                fixed = np.logical_or(left_boundary, right_boundary)
            else:
                # target_x[left_boundary, 0] -= v * (t2-t1)
                # fixed = left_boundary
                fixed = np.array([False] * target_x.shape[0])
            return fixed, target_x
        settings['dirichlet'] = dirichlet
        adjust_camera()
        settings['mesh_scale'] *= 0.1
        settings['mesh_offset'] += [0.35, 0.5]
        settings['boundary'] = find_boundary(settings['mesh_elements'])
        return settings

