import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh

# Add imports if needed:
"""
Your code here
"""


# end imports


# Add functions here:
def points4to8(p1):
    mid_points = np.array([(p1[0] + p1[1]) / 2,
                           (p1[1] + p1[2]) / 2,
                           (p1[2] + p1[3]) / 2,
                           (p1[3] + p1[0]) / 2])
    return np.concatenate((p1, mid_points), axis=0)
# Functions end


# HW functions:
def create_ref(im_path):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    # mark points from top left clockwise
    book_corners = np.array(plt.ginput(4, timeout=-1))
    plt.close()

    p1 = points4to8(book_corners)

    target_width = int(np.maximum(np.linalg.norm(book_corners[1] - book_corners[0]),
                                  np.linalg.norm(book_corners[2] - book_corners[3])))
    target_height = int(np.maximum(np.linalg.norm(book_corners[2] - book_corners[1]),
                                   np.linalg.norm(book_corners[3] - book_corners[0])))

    target = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])
    target = points4to8(target)

    H = mh.computeH(p1.T, target.T)

    ref_image = cv2.warpPerspective(im, H, (target_width, target_height))
    return ref_image


def im2im(source_ref, dest_im, idx):
    plt.imshow(dest_im)

    # mark points from top left clockwise
    dest_corners = np.array(plt.ginput(4, timeout=-1))
    plt.close()

    dest_p = points4to8(dest_corners)

    source_p = np.array([[0, 0], [source_ref.shape[1], 0],
                       [source_ref.shape[1], source_ref.shape[0]], [0, source_ref.shape[0]]])
    source_p = points4to8(source_p)

    H = mh.computeH(dest_p.T, source_p.T)

    dest_warped, (bx, by) = mh.warpH(dest_im, H)
    np.copyto(dest_warped[bx:source_ref.shape[0] + bx, by:source_ref.shape[1] + by], source_ref)

    plt.imshow(dest_warped)
    plt.show()

    p1, p2 = mh.getPoints_SIFT(dest_im, dest_warped, 0.7, 50)
    H_back = mh.computeH(p2, p1)

    planted_image = cv2.warpPerspective(dest_warped, H_back, (dest_im.shape[1], dest_im.shape[0]))
    return planted_image


if __name__ == '__main__':
    print('my_ar')

    ref_image = create_ref("my_data/book3.jpg")
    plt.imshow(ref_image)
    plt.show()
    cv2.imwrite("my_data/book3_ref.jpg", cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))

    scene = cv2.cvtColor(cv2.imread("my_data/3books.jpg"), cv2.COLOR_BGR2RGB)

    book1_ref = cv2.cvtColor(cv2.imread("my_data/book1_ref.jpg"), cv2.COLOR_BGR2RGB)
    planted_book1 = im2im(book1_ref, scene, 1)
    cv2.imwrite("../output/im2im1.jpg", cv2.cvtColor(planted_book1, cv2.COLOR_RGB2BGR))

    book2_ref = cv2.cvtColor(cv2.imread("my_data/book2_ref.jpg"), cv2.COLOR_BGR2RGB)
    planted_book2 = im2im(book2_ref, scene, 1)
    cv2.imwrite("../output/im2im2.jpg", cv2.cvtColor(planted_book2, cv2.COLOR_RGB2BGR))

    book3_ref = cv2.cvtColor(cv2.imread("my_data/book3_ref.jpg"), cv2.COLOR_BGR2RGB)
    planted_book3 = im2im(book3_ref, scene, 1)
    cv2.imwrite("../output/im2im3.jpg", cv2.cvtColor(planted_book3, cv2.COLOR_RGB2BGR))
