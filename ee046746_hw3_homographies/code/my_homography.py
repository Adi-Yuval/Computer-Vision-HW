import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

#Add imports if needed:
from scipy.io import savemat, loadmat
import pickle
#end imports

#Add extra functions here:
STICHED_IMAGE_SCALE = 1.3


def stitch_our_points(im1, im2, p1, p2):
    H2to1 = computeH(p1, p2)

    interp_method = cv2.INTER_CUBIC
    im1_warp, warped_image_translation = warpH(im1, H2to1, interp_method)

    stitched = imageStitching(im2, im1_warp, warped_image_translation)

    return stitched


def stitch_sift(im1, im2, threshold=0.4, k_matches=None, mask1=None, mask2=None):
    p1, p2 = getPoints_SIFT(im1, im2, threshold, k_matches, mask1, mask2)

    H2to1 = computeH(p1, p2)
    interp_method = cv2.INTER_CUBIC
    im1_warp, warped_image_translation = warpH(im1, H2to1, interp_method)

    stitched = imageStitching(im2, im1_warp, warped_image_translation)
    return stitched


def stitch_incline_our_points():
    im1 = cv2.cvtColor(cv2.imread('data/incline_L.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/incline_R.png'), cv2.COLOR_BGR2RGB)

    # p1, p2 = getPoints(im1, im2, 8)
    points_mat = loadmat('my_data/homography_points.mat')
    p1 = points_mat['p1']
    p2 = points_mat['p2']

    stitched = stitch_our_points(im1, im2, p1, p2)
    plt.imshow(stitched)
    plt.show()


def stitch_incline_sift():
    im1 = cv2.cvtColor(cv2.imread('data/incline_L.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/incline_R.png'), cv2.COLOR_BGR2RGB)

    stitched = stitch_sift(im1, im2)
    plt.imshow(stitched)
    plt.show()

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
    warp_im1 = cv2.warpPerspective(im1, H, dsize=(bwidth, bheight), flags=interp_method)

    return warp_im1, (-by, -bx)


def imageStitching(img1, wrap_img2, warped_image_translation):
    panoImg = np.zeros((img1.shape[0] + wrap_img2.shape[0],
                        img1.shape[1] + wrap_img2.shape[1],
                        3),
                       dtype=np.uint8)

    # if im2 was translated right or down, then image 1 has to be translated there too,
    # if it was translated left or up, it should be translated back
    if warped_image_translation[1] >= 0:
        im2_x = 0
        im1_x = warped_image_translation[1]
    else:
        im2_x = -warped_image_translation[1]
        im1_x = 0

    if warped_image_translation[0] >= 0:
        im2_y = 0
        im1_y = warped_image_translation[0]
    else:
        im2_y = -warped_image_translation[0]
        im1_y = 0

    panoImg[im1_y:im1_y + img1.shape[0], im1_x:im1_x + img1.shape[1]] =\
        np.maximum(img1, panoImg[im1_y:im1_y + img1.shape[0], im1_x:im1_x + img1.shape[1]])
    panoImg[im2_y:im2_y + wrap_img2.shape[0], im2_x:im2_x + wrap_img2.shape[1]] =\
        np.maximum(wrap_img2, panoImg[im2_y:im2_y + wrap_img2.shape[0], im2_x:im2_x + wrap_img2.shape[1]])

    # clear black space:
    max_y = max(im1_y + img1.shape[0], im2_y + wrap_img2.shape[0])
    max_x = max(im1_x + img1.shape[1], im2_x + wrap_img2.shape[1])
    panoImg = panoImg[:max_y, :max_x]

    return panoImg

def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    return bestH

def getPoints_SIFT(im1, im2, dist_thresh=0.4, k_matches=None, mask1=None, mask2=None):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, mask1)
    kp2, des2 = sift.detectAndCompute(im2, mask2)

    # index_params = dict(algorithm=1, trees=5)
    # search_params = dict(checks=50)
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des1, des2, k=2)

    matches = sorted(matches, key=lambda x: x[0].distance)

    p1 = []
    p2 = []
    for m, n in matches:
        if k_matches and len(p1) >= k_matches:
            break
        if m.distance < dist_thresh * n.distance:
            p1.append(kp1[m.queryIdx].pt)
            p2.append(kp2[m.trainIdx].pt)

    return np.array(p1).T, np.array(p2).T


if __name__ == '__main__':
    print('my_homography')

    # stitch_incline_our_points()
    # stitch_incline_sift()

    # load beach images:
    beach_images = []
    for i in range(1, 6):
        im = cv2.imread('data/beach' + str(i) + '.jpg')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (im.shape[0] // 2, im.shape[1] // 2))
        beach_images.append(im)
    beach_images.reverse()


    # stitched_beach = beach_images[0]
    # for i in range(1, len(beach_images)):
    #     stitched_beach = stitch_sift(stitched_beach, beach_images[i], threshold=0.7, k_matches=25)
    #     plt.imshow(stitched_beach)
    #     plt.show()

    # stitched_01 = stitch_sift(beach_images[0], beach_images[1], threshold=0.7, k_matches=25)
    # plt.imshow(stitched_01)
    # plt.show()
    #
    # stitched_10 = stitch_sift(beach_images[1], beach_images[0], threshold=0.7, k_matches=25)
    # plt.imshow(stitched_10)
    # plt.show()

    #
    stitched_beach_up = stitch_sift(beach_images[0], beach_images[1], threshold=0.5, k_matches=50)
    stitched_beach_up = stitch_sift(stitched_beach_up, beach_images[2], threshold=0.5, k_matches=50)
    # plt.imshow(stitched_beach_up)
    # plt.show()
    stitched_beach_down = stitch_sift(beach_images[4], beach_images[3], threshold=0.5, k_matches=50)
    stitched_beach_down = stitch_sift(stitched_beach_down, beach_images[2], threshold=0.5, k_matches=50)
    # plt.imshow(stitched_beach_down)
    # plt.show()

    # mask1 = np.ones((stitched_beach_up.shape[0], stitched_beach_up.shape[1]), dtype=np.uint8)
    # mask1[:stitched_beach_up.shape[0]//2, :] = 0
    # mask2 = np.ones((stitched_beach_down.shape[0], stitched_beach_down.shape[1]), dtype=np.uint8)
    # mask2[stitched_beach_down.shape[0]//3:, :] = 0
    # #
    stitched_beach = stitch_sift(stitched_beach_up, stitched_beach_down, threshold=0.5, k_matches=50)
    plt.imshow(stitched_beach)
    plt.show()

    # points_pkl = pickle.load(open('data/points.pkl', 'rb'))
    # beach_points = points_pkl['beach']
    #
    #
    # pass