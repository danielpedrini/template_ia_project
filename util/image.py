def img_filtering(img, filter):
    import cv2
    import numpy as np
    
    identity = np.array(([0, 0, 0],[0, 1, 0], [0, 0, 0]), np.float32)
    edge_1 = np.array(([0, -1, 0],[-1, 4, -1],[0, -1, 0]), np.float32)
    edge_2 = np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]), np.float32)
    sharpen = np.array(([0, -1, 0],[-1, 5, -1], [0, -1, 0]), np.float32)
    box_blur = np.array(([1, 1, 1],[1, 1, 1], [1, 1, 1]), np.float32)/9
    gaussing_blur_3 = np.array(([1, 2, 1],[2, 4, 2], [1, 2, 1]), np.float32)/16
    gaussing_blur_5 = np.array(([1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]), np.float32)/256

    if filter == 'edge_1':
        k = edge_1
    elif filter == 'edge_2':
        k = edge_2
    elif filter == 'sharpen':
        k = sharpen
    elif filter == 'box_blur':
        k = box_blur
    elif filter == 'gaussing_blur_3':
        k = gaussing_blur_3
    elif filter == 'gaussing_blur_5':
        k = gaussing_blur_5
    else:
        k = identity

    filtered_img = cv2.filter2D(img, -1, k)
    return filtered_img