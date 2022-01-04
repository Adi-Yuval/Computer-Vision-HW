import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh


#Add imports if needed:
"""
Your code here
"""
#end imports


#Add functions here:
"""
   Your code here
"""
#Functions end


# HW functions:
def create_ref(im_path):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    # mark points from top left clockwise
    book_corners = np.array(plt.ginput(8, timeout=-1))
    plt.close()

    target_width = int(np.maximum(np.linalg.norm(book_corners[1] - book_corners[0]),
                                  np.linalg.norm(book_corners[2] - book_corners[3])))
    target_height = int(np.maximum(np.linalg.norm(book_corners[2] - book_corners[1]),
                                   np.linalg.norm(book_corners[3] - book_corners[0])))

    target_corners = np.array([[0, 0], [target_width//2, 0], [target_width, 0], [target_width, target_height//2],
                               [target_width, target_height], [target_width//2, target_height], [0, target_height], [0, target_height//2]])

    H = mh.computeH(book_corners.T, target_corners.T)
    # H = np.linalg.inv(H)

    ref_image, (bx, by) = mh.warpH(im, H)
    ref_image = ref_image[bx:target_height+bx, by:target_width+by, :]
    return ref_image

if __name__ == '__main__':
    print('my_ar')

    ref_image = create_ref("my_data/book_cover_img.jpg")
    plt.imshow(ref_image)
    plt.show()
    cv2.imwrite("my_data/book_cover_ref.jpg", cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
