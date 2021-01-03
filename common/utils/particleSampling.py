import numpy as np
import triangle as tr
import matplotlib.pyplot as plt

#Sample from TetWild tetrahedral mesh
def sampleFromTetWild(filename, density):
    X = []
    idx = []
    readTetWildFile(filename, X, idx)

    #Now compute the total volume
    #Method from this link: https://stackoverflow.com/questions/9866452/calculate-volume-of-any-tetrahedron-given-4-points
    volume = 0.0
    for i in range(len(idx)): #iterate over tets
        a = X[idx[i][0]]
        b = X[idx[i][1]]
        c = X[idx[i][2]]
        d = X[idx[i][3]]
        A = np.matrix([[a[0], b[0], c[0], d[0]], [a[1], b[1], c[1], d[1]], [a[2], b[2], c[2], d[2]], [1.0, 1.0, 1.0, 1.0]])
        currVol = np.linalg.det(A) / 6.0
        if currVol < 0: currVol *= -1
        volume += currVol

    print("[Particle Sampling] Total TetMesh Volume: ", volume)
    return np.array(X), float(volume)

#Read from TetWild tet mesh to get points and indeces
def readTetWildFile(filename, samples, indeces):
    readingPoints = False
    readingTriangles = False
    readingTets = False
    numPoints = 0
    numTris = 0
    numTets = 0
    try:
        f = open(filename)
        for line in f:
            if line[:4] == "Mesh":
                continue
            elif line[:9] == "Dimension":
                continue
            elif line[:8] == "Vertices":
                readingPoints = True
                readingTriangles = False
                readingTets = False
                continue
            elif line[:9] == "Triangles":
                readingPoints = False
                readingTriangles = True
                readingTets = False
                continue
            elif line[:10] == "Tetrahedra":
                readingPoints = False
                readingTriangles = False
                readingTets = True
                continue
            elif line[:3] == "End":
                continue
            elif numPoints == 0 and readingPoints:
                numPoints = int(line)
                #print("numPoints:", numPoints)
                continue
            elif numTris == 0 and readingTriangles:
                numTris = int(line)
                #print("numTris:", numTris)
                if numTris == 0:
                    readingTriangles = False
                continue
            elif numTets == 0 and readingTets:
                numTets = int(line)
                #print("numTets:", numTets)
                continue
            elif readingPoints:
                index1 = 0
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)
                index4 = line.find(" ", index3 + 1)
                vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:index4]))
                samples.append(vertex)
            elif readingTriangles:
                continue
            elif readingTets:
                index1 = 0
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)
                index4 = line.find(" ", index3 + 1)
                index5 = line.find(" ", index4 + 1)
                idx = (int(line[index1:index2]) - 1, int(line[index2:index3]) - 1, int(line[index3:index4]) - 1, int(line[index4:index5]) - 1) #NOTE: need to subtract by 1 to each index since they start at 1 (*sigh* plebs)
                indeces.append(idx)
        f.close()
        assert(numPoints == len(samples))
        assert(numTets == len(indeces))
    except IOError:
        print("IO ERROR: .mesh file not found")
    return


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

#Read OBJ and then triangulate using Triangle
def readOBJAndTriangulate(filepath, args):
    return 0

#Use Triangle to sample a triangulated square
def sampleBox2D(minPoint, maxPoint, args = 'qa0.0000075'):
    A = dict(vertices=np.array(((minPoint[0], minPoint[1]), (maxPoint[0], minPoint[1]), (maxPoint[0], maxPoint[1]), (minPoint[0], maxPoint[1]))))
    B = tr.triangulate(A, args)
    return np.array(B.get('vertices'))

def sampleNotchedWall2D(minPoint, maxPoint, args = 'qpa0.0000075'):

    height = maxPoint[1] -  minPoint[1]
    width = maxPoint[0] - minPoint[0]
    midX = minPoint[0] + (width / 2.0)
    midY = minPoint[1] + (height / 2.0)

    notchHeight = height * 0.1
    notchDepth = width * 0.2

    notchPoint1 = (maxPoint[0], midY + (notchHeight/2.0))
    notchPoint2 = (maxPoint[0] - notchDepth, midY)
    notchPoint3 = (maxPoint[0], midY - (notchHeight / 2.0))

    pts1 = np.array(((minPoint[0], minPoint[1]), (maxPoint[0], minPoint[1]), (maxPoint[0], maxPoint[1]), (minPoint[0], maxPoint[1])))
    pts2 = np.array((notchPoint1, notchPoint2, notchPoint3))
    pts = np.vstack([pts1, pts2])
    segs1 = np.array(((0,1),(1,2),(2,3),(3,0)))
    segs2 = np.array(((0,1),(1,2),(2,0)))
    segs = np.vstack([segs1, segs2 + segs1.shape[0]])
    #A = dict(vertices=pts1, segments=segs1)
    A = dict(vertices=pts, segments=segs, holes=[[maxPoint[0] - (notchDepth * 0.1) , midY]])
    B = tr.triangulate(A, args)
    #tr.compare(plt, A, B)
    #plt.show()

    return np.array(B.get('vertices'))


def sampleNotchedBox2D(minPoint, maxPoint, args = 'qa0.0000075'):
    height = maxPoint[1] -  minPoint[1]
    width = maxPoint[0] - minPoint[0]
    #midX = minPoint[0] + (width / 2.0)
    midY = minPoint[1] + (height / 2.0)

    notchHeight = height * 0.1
    notchDepth = width * 0.2

    notchPoint1 = (minPoint[0], midY + (notchHeight/2.0))
    notchPoint2 = (minPoint[0] + notchDepth, midY)
    notchPoint3 = (minPoint[0], midY - (notchHeight / 2.0))

    pts1 = np.array(((minPoint[0], minPoint[1]), (maxPoint[0], minPoint[1]), (maxPoint[0], maxPoint[1]), (minPoint[0], maxPoint[1])))
    #pts2 = np.array((notchPoint1, notchPoint2, notchPoint3))
    #pts = np.vstack([pts1, pts2])
    segs1 = np.array(((0,1),(1,2),(2,3),(3,0)))
    #segs2 = np.array(((0,1),(1,2),(2,0)))
    #segs = np.vstack([segs1, segs2 + segs1.shape[0]])
    A = dict(vertices=pts1, segments=segs1)
    #A = dict(vertices=pts, segments=segs, holes=[[minPoint[0] + 0.015 , midY]])
    #args = 'qa0.0000075'
    B = tr.triangulate(A, args)
    #tr.compare(plt, A, B)
    #plt.show()
    
    #Occam's Razor wins the day again as uzhhh
    #Now remove points in the wedge using y = mx + b
    m1 = (notchPoint2[1] - notchPoint1[1]) / (notchPoint2[0] - notchPoint1[0])
    m2 = -m1
    b1 = notchPoint2[1] - (m1 * notchPoint2[0])
    b2 = notchPoint2[1] - (m2 * notchPoint2[0])

    vertices = []
    oldVertices = B.get('vertices')

    for p in oldVertices:
        y1 = m1 * p[0] + b1
        y2 = m2 * p[0] + b2

        if p[1] < y1 and p[1] > y2:
            continue    
        vertices.append(p)

    return np.array(vertices)

def sampleTriangle2D(p1, p2, p3, args = 'qa0.0000075'):
    A = dict(vertices=np.array((p1, p2, p3)))
    #args = 'qa0.0000075'
    B = tr.triangulate(A, args)

    return np.array(B.get('vertices'))

#Use Triangle to sample a triangulated circle
def sampleCircle2D(centerPoint, radius, N, args = 'qa0.0000075'):
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.stack([centerPoint[0] + radius * np.cos(theta), centerPoint[1] + radius * np.sin(theta)], axis=1)
    A = dict(vertices=pts)    
    B = tr.triangulate(A, args)
    return np.array(B.get('vertices'))

def sampleRing2D(centerPoint, r1, r2, N1, N2, args = 'qpa0.0000075'):

    i = np.arange(N1)
    theta1 = i * 2 * np.pi / N1
    pts1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1) * r1
    seg1 = np.stack([i, i + 1], axis=1) % N1

    i = np.arange(N2)
    theta2 = i * 2 * np.pi / N2
    pts2 = np.stack([np.cos(theta2), np.sin(theta2)], axis=1) * r2
    seg2 = np.stack([i, i + 1], axis=1) % N2

    pts = np.vstack([pts1, pts2])
    seg = np.vstack([seg1, seg2 + seg1.shape[0]])

    A = dict(vertices=pts, segments=seg, holes=[[0, 0]])
    B = tr.triangulate(A, args)

    # tr.compare(plt, A, B)
    # plt.show()

    verts = B.get('vertices')

    for p in verts:
        p += centerPoint

    return np.array(verts)


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

#Analytic Box Grid Particle Sample
def sampleBoxGrid3D(minPoint, maxPoint, N):
    dX = maxPoint[0] - minPoint[0]
    dY = maxPoint[1] - minPoint[1]
    dZ = maxPoint[2] - minPoint[2]
    xDiff = dX / float(N)
    yDiff = dY / float(N)
    zDiff = dZ / float(N)
    positions = []
    #positions.append(minPoint)
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                vertex = np.array((minPoint[0] + (xDiff * i), minPoint[1] + (yDiff * j), minPoint[2] + (zDiff * k)))
                positions.append(vertex)

    return np.array(positions)

#Translated Box computed with Triangle
def sampleTranslatedBox2D(minPoint, maxPoint, N, theta, dx, dy, args = 'qa0.0000075'):
    dX = maxPoint[0] - minPoint[0]
    dY = maxPoint[1] - minPoint[1]
    newMin = [0 - dX/2.0, 0 - dY/2.0]
    newMax = [0 + dX/2.0, 0 + dY/2.0]

    A = dict(vertices=np.array(((newMin[0], newMin[1]), (newMax[0], newMin[1]), (newMax[0], newMax[1]), (newMin[0], newMax[1]))))
    B = tr.triangulate(A, args)
    x = B.get('vertices')

    theta_rad = np.radians(theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    r = np.array(((c, -s),(s, c )))
    positions = []
    for p in x:
        updatedVertex = r.dot(p)

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