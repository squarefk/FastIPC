import numpy as np
import triangle as tr

#Read OBJ file and return list of particle positions
def readOBJ(filepath):
    positions = []
    try:
        f = open(filepath)
        for line in f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)

                vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                positions.append(vertex)

            # elif line[0] == "f":
            #     string = line.replace("//", "/")
            #     ##
            #     i = string.find(" ") + 1
            #     face = []
            #     for item in range(string.count(" ")):
            #         if string.find(" ", i) == -1:
            #             face.append(string[i:-1])
            #             break
            #         face.append(string[i:string.find(" ", i)])
            #         i = string.find(" ", i) + 1
            #     ##
            #     self.faces.append(tuple(face))

        f.close()
    except IOError:
        print(".obj file not found.")

    return np.array(positions)

#Use Triangle to sample a triangulated square
def sampleBox2D(minPoint, maxPoint, maxArea):
    A = dict(vertices=np.array(((minPoint[0], minPoint[1]), (maxPoint[0], minPoint[1]), (maxPoint[0], maxPoint[1]), (minPoint[0], maxPoint[1]))))
    args = 'qa' + str(maxArea)
    B = tr.triangulate(A, args)

    return np.array(B.get('vertices'))

def sampleTriangle2D(p1, p2, p3):
    A = dict(vertices=np.array((p1, p2, p3)))
    args = 'qa0.0000075'
    B = tr.triangulate(A, args)

    return np.array(B.get('vertices'))

#Use Triangle to sample a triangulated circle
def sampleCircle2D(centerPoint, radius, N, maxArea):
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.stack([centerPoint[0] + radius * np.cos(theta), centerPoint[1] + radius * np.sin(theta)], axis=1)
    A = dict(vertices=pts)    
    #args = 'qa' + str(maxArea)
    args = 'qa0.0000075'
    B = tr.triangulate(A, args)

    return np.array(B.get('vertices'))

#Analytic Box Grid Particle Sample
def sampleBoxGrid2D(minPoint, maxPoint, N, theta, dx, dy):
    dX = maxPoint[0] - minPoint[0]
    dY = maxPoint[1] - minPoint[1]
    xDiff = dX / float(N)
    yDiff = dY / float(N)
    newMin = [0 - dX/2.0, 0 - dY/2.0]
    newMax = [0 + dX/2.0, 0 + dY/2.0]
    theta_rad = np.radians(theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    r = np.array(((c, -s),(s, c )))
    positions = []
    #positions.append(minPoint)
    for i in range(N+1):
        for j in range(N+1):
            vertex = np.array((newMin[0] + (xDiff * i), newMin[1] + (yDiff * j)))

            updatedVertex = r.dot(vertex)

            updatedVertex[0] += dx
            updatedVertex[1] += dy

            positions.append(updatedVertex)

    np_pos = np.array(positions)

    return np_pos

#Analytic Half Box (Ramp) Particle Sample
def sampleRamp2D(minPoint, maxPoint, N):
    dX = maxPoint[0] - minPoint[0]
    dY = maxPoint[1] - minPoint[1]
    xDiff = dX / float(N)
    yDiff = dY / float(N)
    increment = min(xDiff, yDiff) #want uniform particles regardless of ratio of w to h
    m = -yDiff / xDiff
    b = maxPoint[1]
    positions = []
    i = 0
    j = 0
    while minPoint[0] + (increment * i) <= maxPoint[0]:
        while minPoint[1] + (increment * j) <= maxPoint[1]:
            
            vertex = np.array((minPoint[0] + (increment * i), minPoint[1] + (increment * j)))

            #only keep particles above y=mx+b (m = yDiff/xDiff, b=0)
            if vertex[1] >= ((vertex[0] * m) + b):
                j += 1
                continue

            positions.append(vertex)

            j += 1 #update y index
        j = 0
        i += 1 #update x index

    np_pos = np.array(positions)

    return np_pos