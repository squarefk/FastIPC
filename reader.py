import pymesh
import numpy as np


def read(testcase):
    if testcase == 0:
        mesh = pymesh.load_mesh("input/Sharkey.obj")
        dirichlet = np.array([i for i in range(12)])
        mesh_scale= 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.6
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 1.0
        return mesh, dirichlet, mesh_scale, mesh_offset
    elif testcase == 1:
        mesh = pymesh.load_mesh("input/cubes.obj")
        dirichlet = np.array([0, 1])
        mesh_scale= 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.6
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 0.9
        mesh_scale, mesh_offset = 0.6, 0.4
        return mesh, dirichlet, mesh_scale, mesh_offset
    elif testcase == 2:
        mesh = pymesh.load_mesh("input/Sharkey_floor.obj")
        dirichlet = np.array([954, 955])
        mesh_scale= 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 1
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 1
        return mesh, dirichlet, mesh_scale, mesh_offset
    elif testcase == 3:
        mesh = pymesh.load_mesh("input/spheres.obj")
        dirichlet = np.array([0])
        mesh_scale, mesh_offset = 1, 0.4
        return mesh, dirichlet, mesh_scale, mesh_offset
