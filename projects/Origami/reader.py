import meshio
import math
import numpy as np
from common.math.graph_tools import *
from scipy.spatial.transform import Rotation
import os, sys
import json

settings = {}

def color_faces(elements, edges):
    face_id = {}
    for i in range(len(elements)):
        face_id[(elements[i, 0], elements[i, 1], elements[i, 2])] = i
        face_id[(elements[i, 1], elements[i, 2], elements[i, 0])] = i
        face_id[(elements[i, 2], elements[i, 0], elements[i, 1])] = i
    
    opposite_face_id = {} # halfedge -> face id
    edge_type = {}
    for i in range(len(edges)):
        edge1 = (edges[i, 0], edges[i, 1])
        edge_type[edge1] = edges[i, 4]
        oppo = (edges[i, 0], edges[i, 3], edges[i, 1])
        if oppo in face_id:
            opposite_face_id[edge1] = face_id[oppo]
        edge2 = (edges[i, 1], edges[i, 0])
        edge_type[edge2] = edges[i, 4]
        oppo = (edges[i, 0], edges[i, 1], edges[i, 2])
        if oppo in face_id:
            opposite_face_id[edge2] = face_id[oppo]

    color = {0: -1}
    queue = [0]
    while len(queue) > 0:
        index = queue.pop(0)
        assert(index in color)
        current_color = color[index]
        edge_li = [(elements[index, 0], elements[index, 1]), (elements[index, 1], elements[index, 2]), (elements[index, 2], elements[index, 0])]
        for edge in edge_li:
            if edge in opposite_face_id: # has opposite face
                oppo_id = opposite_face_id[edge]
                if oppo_id not in color: # find a uncolored face
                    queue.append(oppo_id)
                    if edge_type[edge] == 0:
                        color[oppo_id] = current_color
                    else:
                        color[oppo_id] = -current_color
    color_list = []
    for i in range(len(elements)):
        color_list.append(color[i])
    return color_list


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
    settings["mesh_face_colors"] = np.array(color_faces(settings["mesh_elements"], settings["mesh_edges"]), dtype=np.int32)

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


def read(testcase):
    ################################################### GENERAL ##################################################
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

        settings['dirichlet'] = fixed
        # settings['mesh_scale'] *= 0.1
        # settings['mesh_offset'] += [0.35, 0.5, 0.5]
        return settings

    elif testcase == 1006:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/wing2.fold")
        x = settings['mesh_particles']
        elem = settings['mesh_elements'][0]
        fixed = np.array([False] * x.shape[0])
        fixed[elem[0]] = True
        fixed[elem[1]] = True
        fixed[elem[2]] = True
        settings['dirichlet'] = fixed
        four_vertices = [[21, 17, 23, 19, 15], [19, 14, 21, 18, 20], [20, 13, 19, 22, 10]]
        settings['four_vertices'] = np.array(four_vertices, dtype=np.int32)
        return settings
    
    elif testcase == 1007:
        init(3)
        settings['gravity'] = 0
        parse_fold("input/wing_clean.fold")
        x = settings['mesh_particles']
        fixed = np.array([False] * x.shape[0])
        fixed[0] = True
        fixed[1] = True
        fixed[11] = True
        fixed[14] = True
        settings['dirichlet'] = fixed
        four_vertices = [[14, 9, 11, 1, 12], [12, 8, 14, 2, 13], [13, 7, 12, 3, 4]]
        settings['four_vertices'] = np.array(four_vertices, dtype=np.int32)
        settings['inner_vertex'] = 5
        settings['outer_vertex'] = 4
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
