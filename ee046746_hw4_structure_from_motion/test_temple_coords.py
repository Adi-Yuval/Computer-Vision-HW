import cv2
import numpy as np
import scipy.optimize
import numpy.linalg as la
import matplotlib.pyplot as plt


def eight_point(pts1, pts2, pmax):
    """
    Eight Point Algorithm
    [I] pts1, points in image 1 (Nx2 matrix)
        pts2, points in image 2 (Nx2 matrix)
        pmax, scalar value computed as max(H1,W1)
    [O] F, the fundamental matrix (3x3 matrix)
    """
    T_norm = np.diag([1/pmax, 1/pmax, 1])
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    pts1_normalized = (T_norm @ pts1_hom.T).T
    pts2_normalized = (T_norm @ pts2_hom.T).T

    A_col_0to2 = np.expand_dims(pts1_normalized[:, 0], 1) * pts2_normalized
    A_col_3to5 = np.expand_dims(pts1_normalized[:, 1], 1) * pts2_normalized
    A_col_6to8 = np.expand_dims(pts2_normalized[:, 2], 1) * pts2_normalized

    A = np.concatenate((A_col_0to2, A_col_3to5, A_col_6to8), axis=1)

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F to have rank 2
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V

    F = refineF(F, pts1_normalized[:, :-1], pts2_normalized[:, :-1])

    # denormalize
    F = T_norm.T @ F @ T_norm

    return F


# helper function 1: singualrizes F using SVD
def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))

    return F


# helper function 2.1: defines an objective function using F and the epipolar constraint
def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))

    return r


# helper function 2.2: refines F using the objective from above and local optimization
def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000
    )

    return _singularize(f.reshape([3, 3]))


def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def epipolar_correspondences(I1, I2, F, pts1, patch_size=9, npts=200):
    """
    Epipolar Correspondences
    [I] I1, image 1 (H1xW1 matrix)
        I2, image 2 (H2xW2 matrix)
        F, fundamental matrix from image 1 to image 2 (3x3 matrix)
        pts1, points in image 1 (Nx2 matrix)
    [O] pts2, points in image 2 (Nx2 matrix)
    """
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    epipolar_lines = (F @ pts1_hom.T).T

    pts2 = []
    for idx, line in enumerate(epipolar_lines):
        a, b, c = line

        pts1_x = pts1[idx, 0]

        x_points = np.linspace(pts1_x-100, pts1_x+100, npts)
        x_points = np.floor(x_points).astype(int)
        y_points = -(a*x_points + c) / b
        y_points = np.floor(y_points).astype(int)

        # clip points to image size:
        y_points = np.clip(y_points, patch_size//2 + 1, I2.shape[0] - 1 - patch_size//2)
        x_points = np.clip(x_points, patch_size//2 + 1, I2.shape[1] - 1 - patch_size//2)

        I1_patch = I1[pts1[idx, 1] - (patch_size-1)//2 : pts1[idx, 1] + (patch_size+1)//2,
                      pts1[idx, 0] - (patch_size-1)//2 : pts1[idx, 0] + (patch_size+1)//2]

        patches_distance_min = np.inf
        x_min, y_min = -1, -1
        for x, y in zip(x_points, y_points):
            I2_patch = I2[y - (patch_size-1)//2 : y + (patch_size+1)//2, x - (patch_size-1)//2 : x + (patch_size+1)//2]

            patches_distance = np.sum((I1_patch - I2_patch)**2)
            if patches_distance < patches_distance_min:
                patches_distance_min = patches_distance
                x_min = x
                y_min = y

        pts2.append([x_min, y_min])

    pts2 = np.array(pts2)
    return pts2


def essential_matrix(F, K1, K2):
    """
    Essential Matrix
    [I] F, the fundamental matrix (3x3 matrix)
        K1, camera matrix 1 (3x3 matrix)
        K2, camera matrix 2 (3x3 matrix)
    [O] E, the essential matrix (3x3 matrix)
    """
    E = K2.T @ F @ K1
    return E


def triangulate(M1, pts1, M2, pts2):
    """
    Triangulation
    [I] M1, camera projection matrix 1 (3x4 matrix)
        pts1, points in image 1 (Nx2 matrix)
        M2, camera projection matrix 2 (3x4 matrix)
        pts2, points in image 2 (Nx2 matrix)
    [O] pts3d, 3D points in space (Nx3 matrix)
    """
    pts3d = []

    A = []
    for pt1, pt2 in zip(pts1, pts2):
        x1, y1 = pt1
        x2, y2 = pt2
        A.append(y1 * M1[2, :] - M1[1, :])
        A.append(M1[0, :] - x1 * M1[2, :])
        A.append(y2 * M2[2, :] - M2[1, :])
        A.append(M2[0, :] - x2 * M2[2, :])

        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        pts3d.append(V[-1, :-1] / V[-1, -1])

        A = []

    pts3d = np.array(pts3d)
    return pts3d


# helper function 5: returns the 4 options for camera matrix M2 given the essential matrix
def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)

    return M2s


def determine_best_M2(M1, ex2_canidates, K2, pts1, pts2):
    """
    Determine the best M2 matrix
    [I] ex2_canidates, a list of possible ex2 matrices
    [O] best_M2, the best M2 matrix (3x4 matrix)
    """
    best_M2 = None
    best_behind_camera = 0
    for i in range(4):
        ex2 = ex2_canidates[:, :, i]
        M2 = K2 @ ex2

        pts3d = triangulate(M1, pts1, M2, pts2)

        pts3d_homo = np.concatenate((pts3d, np.ones((pts3d.shape[0], 1))), axis=1)

        behind_camera2 = np.sum((ex2 @ pts3d_homo.T).T[:, 2] < 0)
        behind_camera1 = np.sum(pts3d[:, 2] < 0)

        behind_camera = behind_camera2 + behind_camera1
        if behind_camera <= best_behind_camera:
            best_M2 = M2
            best_behind_camera = behind_camera

    return best_M2


if __name__ == "__main__":

    im1 = cv2.cvtColor(cv2.imread('data/im1.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/im2.png'), cv2.COLOR_BGR2RGB)

    data = np.load("data/some_corresp.npz")
    pts1 = data["pts1"]
    pts2 = data["pts2"]

    F = eight_point(pts1, pts2, pmax=max(im1.shape[0], im1.shape[1]))

    tempale_coords_im1 = np.load("data/temple_coords.npz")['pts1']
    tempale_coords_im2 = epipolar_correspondences(im1, im2, F, tempale_coords_im1, patch_size=17, npts=200)

    intrinsics = np.load('data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    E = essential_matrix(F, K1, K2)

    M1 = K1 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    ex2_canidates = camera2(E)
    M2 = determine_best_M2(M1, ex2_canidates, K2, pts1, pts2)

    # compute the 3D points
    pts3d = triangulate(M1, tempale_coords_im1, M2, tempale_coords_im2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:, 2], -pts3d[:, 0], -pts3d[:, 1], s=30)

    plt.show()