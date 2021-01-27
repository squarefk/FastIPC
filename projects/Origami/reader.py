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
    edges = {}
    mesh_elements = data["faces_vertices"]
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
        edge_types[tuple(edge)] = 0 if t == "B" or t == "F" else 1 if t == "M" else -1
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
    # settings["mesh_edges"] = settings["mesh_edges"][settings["mesh_edges"][:, 3] >= 0]

def init(dim):
    settings['dim'] = dim
    settings['mesh_particles'] = np.zeros((0, dim), dtype=np.float64)
    settings['mesh_elements'] = np.zeros((0, dim + 1), dtype=np.int32)
    settings['mesh_scale'] = 1.
    settings['mesh_offset'] = [0., 0.] if dim == 2 else [0., 0., 0.]
    settings['mesh_edges'] = np.zeros((0, 5), dtype=np.int32) 
        # [v0, v1, v2, v3, t] where (v0 v1) is the edge, v2 v3 are two vertices beside the edge. 
        # v0 v1 v2 are in the counter-clockwise order.
        # t is crease type: 0 - inner/boundary edges; 1 - M; -1 - V


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
    if testcase == 1001:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/crane.fold")
        def dirichlet(t):
            x = settings['mesh_particles']
            elem = settings['mesh_elements'][0]
            fixed = np.array([False] * x.shape[0])
            fixed[elem[0]] = True
            fixed[elem[1]] = True
            fixed[elem[2]] = True
            return fixed, x.copy()

        def rest_angle(t):
            types = settings['mesh_edges'][:, 4]
            if t < 2:
                rest_angle = 179. / 2. * t * types
            else:
                rest_angle = 179 * types
            return rest_angle

        settings['dirichlet'] = dirichlet
        settings['rest_angle'] = rest_angle
        adjust_camera()
        # settings['mesh_scale'] *= 0.1
        # settings['mesh_offset'] += [0.35, 0.5]
        return settings

    elif testcase == 1002:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/simple_fold.fold")
        x = settings['mesh_particles']
        elem = settings['mesh_elements'][0]
        fixed = np.array([False] * x.shape[0])
        # fixed[elem[0]] = True
        # fixed[elem[1]] = True
        # fixed[elem[2]] = True

        def rest_angle(t):
            types = settings['mesh_edges'][:, 4]
            if t < 4:
                rest_angle = np.pi * 0.999 / 4. * t * types
            else:
                rest_angle = np.pi * 0.999 * types
            return rest_angle
            # return np.zeros_like(types)

        settings['dirichlet'] = fixed
        settings['rest_angle'] = rest_angle
        adjust_camera()
        # settings['mesh_scale'] *= 0.1
        # settings['mesh_offset'] += [0.35, 0.5, 0.5]
        return settings

    elif testcase == 1003:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/flappingBird.fold")
        # settings["mesh_particles"] *= 0.1
        x = settings['mesh_particles']
        elem = settings['mesh_elements'][20]
        fixed = np.array([False] * x.shape[0])
        fixed[elem[0]] = True
        fixed[elem[1]] = True
        fixed[elem[2]] = True

        def rest_angle(t):
            types = settings['mesh_edges'][:, 4]
            if t < 4:
                rest_angle = np.pi / 4. * t * types
            else:
                rest_angle = np.pi * types
            return rest_angle

        settings['dirichlet'] = fixed
        settings['rest_angle'] = rest_angle
        adjust_camera()
        # settings['mesh_scale'] *= 0.1
        # settings['mesh_offset'] += [0.35, 0.5]
        return settings
    
    
    elif testcase == 1005:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/wing_cp.fold")
        x = settings['mesh_particles']
        elem = settings['mesh_elements'][0]
        fixed = np.array([False] * x.shape[0])
        fixed[elem[0]] = True
        fixed[elem[1]] = True
        fixed[elem[2]] = True

        def rest_angle(t):
            types = settings['mesh_edges'][:, 4]
            if t < 4:
                rest_angle = np.pi / 4 * t * types
            else:
                rest_angle = np.pi * types
            return rest_angle
            # return np.zeros_like(types)

        settings['dirichlet'] = fixed
        settings['rest_angle'] = rest_angle
        adjust_camera()
        # settings['mesh_scale'] *= 0.1
        # settings['mesh_offset'] += [0.35, 0.5, 0.5]
        return settings

    elif testcase == 1006:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/huffmanWaterbomb.fold")

        def rest_angle(t):
            types = settings['mesh_edges'][:, 4]
            if t < 4:
                rest_angle = np.pi / 4 * t * types
            else:
                rest_angle = np.pi * types
            return rest_angle
            # return np.zeros_like(types)
        x = settings['mesh_particles']
        elem = settings['mesh_elements'][0]
        fixed = np.array([False] * x.shape[0])
        fixed[elem[0]] = True
        fixed[elem[1]] = True
        fixed[elem[2]] = True
        settings['dirichlet'] = fixed
        settings['rest_angle'] = rest_angle
        adjust_camera()
        # settings['mesh_scale'] *= 0.1
        # settings['mesh_offset'] += [0.35, 0.5, 0.5]
        return settings

    # debug
    elif testcase == 10001:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/debug.fold")
        fixed = np.array([False] * settings['mesh_particles'].shape[0])
        def rest_angle(t):
            types = settings['mesh_edges'][:, 4]
            # if t < 4:
            #     rest_angle = 179. / 4. * t * types
            # else:
            #     rest_angle = 179 * types
            # return rest_angle
            return np.zeros_like(types)

        settings['dirichlet'] = fixed
        settings['rest_angle'] = rest_angle
        adjust_camera()
        # settings['mesh_scale'] *= 0.1
        # settings['mesh_offset'] += [0.35, 0.5, 0.5]
        return settings
