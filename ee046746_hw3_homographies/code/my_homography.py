import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

#Add imports if needed:
from scipy.io import savemat, loadmat
#end imports

#Add extra functions here:
"""
Your code here
"""

STICHED_IMAGE_SCALE = 1.3
#Extra functions end

# HW functions:
def getPoints(im1, im2, N):
    plt.imshow(im1)
    p1 = np.array(plt.ginput(N, timeout=-1))
    plt.imshow(im2)
    p2 = np.array(plt.ginput(N, timeout=-1))

    return p1.T, p2.T


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    p1 = p1.T
    p2 = p2.T

    # six first columns of a are built of two sparse p1 matrices
    p1_homo = np.concatenate([p1, np.ones((p1.shape[0], 1))], axis=1)
    p1_homo_sparse = np.zeros(( 2*p1_homo.shape[0], 3))
    p1_homo_sparse[::2] = p1_homo
    p1_homo_sparse_offb1 = np.zeros_like(p1_homo_sparse)
    p1_homo_sparse_offb1[1::2] = p1_homo

    p1_tiled = p1_homo_sparse_offb1 + p1_homo_sparse
    p2_as_col = p2.reshape(-1, 1)
    # tree last columns are built of the following product:
    a_6to8 = - p1_tiled * p2_as_col

    # build A matrix:
    a = np.concatenate((p1_homo_sparse, p1_homo_sparse_offb1, a_6to8), axis=1)

    # find h using SVD:
    U, D, V = np.linalg.svd(a, False)
    h = V.T[:, -1]
    H2to1 = h.reshape(3, 3)

    return H2to1


def warpH(im1, H, interp_method=cv2.INTER_LINEAR):
    # find the corners of the warped image:
    height, width = im1.shape[:2]
    corners = np.array([
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0]
    ])
    corners = cv2.perspectiveTransform(np.float32([corners]), H)[0]

    # find bounding box for corners:
    bx, by, bwidth, bheight = cv2.boundingRect(corners)

    # add translation to H:
    tH = np.array([[1, 0, -bx],
                   [0, 1, -by],
                   [0, 0, 1]],)
    H = tH @ H

    # warp image:
    warp_im1 = cv2.warpPerspective(im1, H, dsize=(int(STICHED_IMAGE_SCALE*im1.shape[1]) - bx,
                                                  int(STICHED_IMAGE_SCALE*im1.shape[0]) - by),
                                   flags=interp_method)

    return warp_im1


def imageStitching(img1, wrap_img2, warped_image_translation):
    img1 = cv2.copyMakeBorder(img1,
                              0,
                              wrap_img2.shape[0] - img1.shape[0],
                              0,
                              wrap_img2.shape[1] - img1.shape[1],
                              cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])

    tH = np.array([[1, 0, warped_image_translation[1]],
                   [0, 1, warped_image_translation[0]],
                   [0, 0, 1]], dtype=np.float32)

    translate_im1 = cv2.warpPerspective(img1, tH, dsize=(img1.shape[1], img1.shape[0]), flags=cv2.INTER_CUBIC)
    panoImg = np.maximum(translate_im1, wrap_img2)
    return panoImg

def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    return bestH

def getPoints_SIFT(im1,im2):
    """
    Your code here
    """
    return p1,p2

if __name__ == '__main__':
    print('my_homography')

    im1 = cv2.cvtColor(cv2.imread('data/incline_L.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/incline_R.png'), cv2.COLOR_BGR2RGB)

    # p1, p2 = getPoints(im1, im2, 8)

    points_mat = loadmat('my_data/homography_points.mat')
    p1 = points_mat['p1']
    p2 = points_mat['p2']

    H2to1 = computeH(p1, p2)

    interp_method = cv2.INTER_CUBIC
    im1_warp = warpH(im1, H2to1, interp_method)

    warped_image_translation = np.array(im1_warp.shape[:2]) - np.array(STICHED_IMAGE_SCALE*np.array(im1.shape[:2]), dtype=np.int)
    stitched = imageStitching(im2, im1_warp, warped_image_translation)



    plt.imshow(stitched)
    plt.show()









