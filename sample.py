import os
import cv2
import numpy as np



# intrinsics of image 1
i1 = "853.333312988 0.0 640.0 0.0 0.0 853.333312988 360.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"

# intrinsics of image 2
i2 = "853.333312988 0.0 640.0 0.0 0.0 853.333312988 360.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"

# c2w of image 1
e1 = "-0.15209077142290153 0.0969374640578288 -0.9836013040404247 0.7977413160901098 -0.027629160068689942 0.9943727796408467 0.10227123067233912 0.0166313253772079 0.9879802765044886 0.04273058824043063 -0.14855662242640463 0.21242934235717686 0.0 0.0 0.0 1.0"

# c2w of image 2
e2 = "-0.06316827764469812 0.10186434483002596 -0.9927907251539729 0.78954613166833 -0.02965661762857 0.9941465017267146 0.10389041406870708 0.0158979260258472 0.9975621553217797 0.03600539344159688 -0.059777570315969586 0.15105480953709308 0.0 0.0 0.0 1.0"

# images
im1 = cv2.imread("africa03200.jpg")
im2 = cv2.imread("africa03240.jpg")


i1 = np.array(list(map(lambda x: float(x), i1.split(" ")))).reshape([4,4])[:3,:3]
i2 = np.array(list(map(lambda x: float(x), i2.split(" ")))).reshape([4,4])[:3,:3]
e1 = np.array(list(map(lambda x: float(x), e1.split(" ")))).reshape([4,4])
e2 = np.array(list(map(lambda x: float(x), e2.split(" ")))).reshape([4,4])

H, W, C = im1.shape

def process_pose(C2W):
    flip_yz = np.eye(4, dtype=C2W.dtype)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def process_intrinsic(intrinsic, H, W):
    intrinsic[0, 2] = intrinsic[0, 2] - W / 2
    intrinsic[1, 2] = intrinsic[1, 2] - H / 2
    intrinsic[1, 2] = - intrinsic[1, 2]
    return intrinsic

i1 = process_intrinsic(i1, H, W)
i2 = process_intrinsic(i2, H, W)

e1 = process_pose(e1)
e2 = process_pose(e2)


def get_ray(i1, e1, coord):
    coord = np.reshape([coord[0], coord[1], -1.], [3,1])
    ray_d = np.matmul(np.linalg.inv(i1[:3, :3]), coord)
    ray_d = np.matmul(e1[:3, :3], ray_d)  # (3 x 1)
    ray_d = np.transpose(ray_d, (1, 0))  # (1 x 3)
    ray_d = np.reshape(ray_d, (3))

    ray_o = np.reshape(e1[:3, 3], (3))

    return ray_o, ray_d

def world2cam(i, c2w, point):
    w2c = np.linalg.inv(c2w)
    point = np.reshape([point[0], point[1], point[2], 1.], [4,1])
    point = np.matmul(w2c, point)
    point = point[0:3] / point[3:4]
    coord = np.matmul(i, point)
    coord = coord[0:2] / -coord[2:3]
    return coord

def draw_line(x1, x2, W, H):
    x1 = np.array(x1)
    x2 = np.array(x2)
    diff = x2 - x1
    if diff[0] == 0.:
        sp = np.array([x1[0], 0])
        ep = np.array([x1[0], H-1])
        return sp, ep
    elif diff[1] == 0.:
        sp = np.array([0., x1[1]])
        ep = np.array([W-1., x1[1]])
        return sp, ep
    else:
        slope_x = diff[1] / diff[0]
        slope_y = diff[0] / diff[1]

        x0y = slope_x * (-x1[0]) + x1[1]
        xwy = slope_x * (W-1.-x1[0]) + x1[1]

        y0x = slope_y * (-x1[1]) + x1[0]
        yhx = slope_y * (H - 1. - x1[1]) + x1[0]

        points = []
        if 0. <= x0y <= H-1:
            p = np.array([0., x0y])
            points.append(p)

        if 0. <= xwy <= H-1:
            p = np.array([W - 1., xwy])
            points.append(p)

        if 0. < y0x < W-1:
            p = np.array([y0x, 0.])
            points.append(p)

        if 0. < yhx < W-1:
            p = np.array([yhx, H-1.])
            points.append(p)

        return points


def click(event, x, y, flags, param):
    global im1, im2
    if event == cv2.EVENT_LBUTTONDOWN:
        H, W, C = im1.shape
        # im2 = ims[1][..., [2, 1, 0]].copy()
        im1 = im1.copy()
        im2 = im2.copy()

        clicked = (x - W//2, H//2 - y)

        ray_o, ray_d = get_ray(i1, e1, clicked)
        #print(ray_o, ray_d)
        near = ray_o
        far = ray_o + ray_d

        n2 = world2cam(i2, e2, near)
        f2 = world2cam(i2, e2, far)


        print(n2, f2)

        start_point = (int(n2[0] + W//2), int(- n2[1] + H//2))
        end_point = (int(f2[0] + W//2), int(- f2[1] + H//2))

        start_point, end_point = draw_line(start_point, end_point, W, H)
        start_point = start_point.astype(np.int32)
        end_point = end_point.astype(np.int32)
        print(start_point, end_point)
        color = np.random.uniform(0, 255, size=3).tolist()
        color = tuple(map(lambda x: float(x), color))
        thickness = 3
        im1 = cv2.circle(im1, (x,y), radius=5, color=color, thickness=-1)
        im2 = cv2.line(im2, start_point, end_point, color, thickness)
        cv2.imshow("im2", im2)


cv2.namedWindow("im1")
cv2.setMouseCallback("im1", click)

while True:
    cv2.imshow("im1", im1)
    cv2.imshow("im2", im2)

    key = cv2.waitKey(1) & 0xFF

    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break

