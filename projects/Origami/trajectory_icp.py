# https://github.com/dakshaau/ICP/blob/master/icp.py
import numpy as np
from scipy.spatial.distance import cdist

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    all_dists = cdist(src, dst, 'euclidean')
    indices = all_dists.argmin(axis=1)
    distances = all_dists[np.arange(all_dists.shape[0]), indices]
    return distances, indices

def icp(A, B, target_A, target_B, init_pose=None, max_iterations=100, tolerance=0.001):
    '''
    The Iterative Closest Point method for two separate clouds
    Input:
        A, B: Nx3 numpy array of source 3D points
        target_A, target_B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''

    # make points homogeneous, copy them so as to maintain the originals
    src1 = np.ones((4,A.shape[0]))
    src2 = np.ones((4,B.shape[0]))
    dst1 = np.ones((4,target_A.shape[0]))
    dst2 = np.ones((4,target_B.shape[0]))
    src1[0:3,:] = np.copy(A.T)
    dst1[0:3,:] = np.copy(target_A.T)
    src2[0:3,:] = np.copy(B.T)
    dst2[0:3,:] = np.copy(target_B.T)

    src_bk = np.concatenate([src1, src2], axis=1)

    # apply the initial pose estimation
    if init_pose is not None:
        src1 = np.dot(init_pose, src1)
        src2 = np.dot(init_pose, src2)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances1, indices1 = nearest_neighbor(src1[0:3,:].T, dst1[0:3,:].T)
        distances2, indices2 = nearest_neighbor(src2[0:3,:].T, dst2[0:3,:].T)

        src = np.concatenate([src1, src2], axis=1)
        dst = np.concatenate([dst1[:, indices1], dst2[:, indices2]], axis=1)
        distances = np.concatenate([distances1, distances2])

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[0:3,:].T, dst[0:3,:].T)

        # update the current source
        src1 = np.dot(T, src1)
        src2 = np.dot(T, src2)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(src_bk[0:3,:].T, src[0:3,:].T)

    return T, indices1, indices2

if __name__ == '__main__':
    import scipy.io
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d 
    from matplotlib import cm
    data = scipy.io.loadmat('input/traj.mat')
    A = data["inner"]
    B = data["outer"]

    data = scipy.io.loadmat('input/wing_tips.mat')
    target_A = data["inner"]
    target_B = data["outer"]


    start_point = 0
    T, distances = icp(A[start_point:], B[start_point:], target_A, target_B)

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    
    X = A[start_point:,0]
    Y = A[start_point:,1]
    Z = A[start_point:,2]
    ax.plot3D(X, Y, Z, 'r') 

    X = B[start_point:,0]
    Y = B[start_point:,1]
    Z = B[start_point:,2]
    ax.plot3D(X, Y, Z, 'r')

    X = target_A[:,0]
    Y = target_A[:,1]
    Z = target_A[:,2]
    ax.plot3D(X, Y, Z, 'g') 

    X = target_B[:,0]
    Y = target_B[:,1]
    Z = target_B[:,2]
    ax.plot3D(X, Y, Z, 'g')

    T = np.linalg.inv(T)

    target_A = (np.dot(T[:3, :3], target_A.T) + T[:3, 3:4]).T
    target_B = (np.dot(T[:3, :3], target_B.T) + T[:3, 3:4]).T

    X = target_A[:,0]
    Y = target_A[:,1]
    Z = target_A[:,2]
    ax.plot3D(X, Y, Z, 'b') 

    X = target_B[:,0]
    Y = target_B[:,1]
    Z = target_B[:,2]
    ax.plot3D(X, Y, Z, 'b')


    set_axes_equal(ax)

    # plt.legend(["origami", "origami", "registered_wing", "registered_wing"])
    # plt.legend(["origami", "origami", "unregistered_wing", "unregistered_wing"])
    plt.legend(["origami", "origami", "unregistered_wing", "unregistered_wing", "registered_wing", "registered_wing"])
    plt.show()