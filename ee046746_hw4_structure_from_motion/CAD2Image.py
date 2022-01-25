import cv2
import numpy as np
import scipy.optimize
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import patches


def estimate_pose(p, P):
    """
    Camera Matrix Estimation
    [I] p, 2D points (Nx2 matrix)
        P, 3D points (Nx3 matrix)
    [O] M, camera matrix (3x4 matrix)
    """  
    P_homo = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    P_homo_sparse = np.zeros((2*P_homo.shape[0], 4))
    P_homo_sparse[::2] = P_homo
    P_homo_sparse_offb1 = np.zeros_like(P_homo_sparse)
    P_homo_sparse_offb1[1::2] = P_homo

    P_tiled = P_homo_sparse_offb1 + P_homo_sparse
    p_as_col = p.reshape(-1, 1)
    # tree last columns are built of the following product:
    a_6to8 = - P_tiled * p_as_col

    # build A matrix:
    a = np.concatenate((P_homo_sparse, P_homo_sparse_offb1, a_6to8), axis=1)

    # find h using SVD:
    U, D, V = np.linalg.svd(a, False)
    m = V.T[:, -1]
    M = m.reshape(3, 4)
    
    return M


def estimate_params(M):
    """
    Camera Parameter Estimation
    [I] M, camera matrix (3x4 matrix)
    [O] K, camera intrinsics (3x3 matrix)
        R, camera extrinsics rotation (3x3 matrix)
        t, camera extrinsics translation (3x1 matrix)
    """
    U, D, V = np.linalg.svd(M, False)
    c = V.T[:, -1]
    c = c[ :3] / c[3]
    N = M[:, :3]
    K, R = scipy.linalg.rq(N)
    t = -R @ c
    return K, R, t


if __name__ == "__main__":
    data = np.load("data/pnp.npz", allow_pickle=True)
    X = data['X']
    x = data['x']
    im = data['image']
    CAD = data['cad']

    M = estimate_pose(x, X)
    K, R, t = estimate_params(M)

    X_homo = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    projected_x_homo = (M @ X_homo.T).T
    projected_x = np.round(projected_x_homo[:, :2] / np.expand_dims(projected_x_homo[:, 2], axis=1)).astype('int32')
    im_for_show = im

    for point in x:
        im_for_show = cv2.circle(cv2.UMat(im_for_show), tuple(np.round(point).astype('int32')), 10, (0, 255, 0), 1).get()
    for point in projected_x:
        im_for_show = cv2.circle(cv2.UMat(im_for_show), tuple(point), 3, (0, 0, 255), -1).get()

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.imshow(im_for_show)

    vertices = CAD[0][0][0]
    rotated_vertices = (R @ vertices.T).T
    faces = CAD[0][0][1]
    ones = np.ones(CAD[0][0][1].shape).astype('uint32')
    faces = faces - ones
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], triangles=faces, color='r', linewidth=0.5, alpha=0.3, shade=False)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_box_aspect((np.ptp(rotated_vertices[:, 0]), np.ptp(rotated_vertices[:, 1]), np.ptp(rotated_vertices[:, 2])))
    ax2.view_init(30, -50)

    vertices_homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    projected_vertices_homo = (M @ vertices_homo.T).T
    projected_vertices = np.round(projected_vertices_homo[:, :2] / np.expand_dims(projected_vertices_homo[:, 2], axis=1)).astype('int32')

    ax3 = fig.add_subplot(133)
    ax3.imshow(im)
    ax3.add_patch(patches.Polygon(np.stack([projected_vertices[:, 0], projected_vertices[:, 1]], 1), alpha=0.3, color='r'))
    plt.show()

    pass

