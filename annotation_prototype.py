import cv2
from tqdm import tqdm
import os
import numpy as np
from shapely.geometry import Polygon, Point

polygon_in_drawing = []
finished_polygons = []


def draw_polygon(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_in_drawing.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        if len(polygon_in_drawing) > 1:
            cv2.line(img, polygon_in_drawing[-2], polygon_in_drawing[-1],
                     (0, 255, 0), 2)


def get_inner_points(polygon):
    inner_points = []
    poly_shape = Polygon(polygon)
    poly_np = np.asarray(polygon)
    x_min = poly_np[:, 0].min()
    x_max = poly_np[:, 0].max()
    y_min = poly_np[:, 1].min()
    y_max = poly_np[:, 1].max()
    for p_x in tqdm(range(x_min, x_max + 1)):
        for p_y in range(y_min, y_max + 1):
            cand_point = Point(p_x, p_y)
            if poly_shape.contains(cand_point):
                inner_points.append((p_y, p_x))
    return np.asarray(inner_points)


def polygons_as_segmask():
    # convert polygons to segmask
    seg_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    for polygon in tqdm(finished_polygons):
        polygon_point_coords = get_inner_points(polygon)
        seg_mask[polygon_point_coords[:, 0], polygon_point_coords[:, 1]] = True
    return seg_mask


save_folder = '/home/nick/segmentations'
input_img_folder = '/home/nick/input_imgs'
input_imgs = os.listdir(input_img_folder)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_polygon)
im_ctr = 0
im_file = os.path.join(input_img_folder, input_imgs[im_ctr])
img = cv2.imread(im_file)

while (1): 
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('f'):
        # finish current polygon
        assert (len(polygon_in_drawing) >=
                3), "Need at least three points to draw polygon"
        cv2.line(img, polygon_in_drawing[-1], polygon_in_drawing[0],
                 (0, 255, 0), 2)
        finished_polygons.append(polygon_in_drawing)
        polygon_in_drawing = []
    elif k == ord('s'):
        # save polygons as segmentation map
        segmask = polygons_as_segmask()
        filename = os.path.join(save_folder, 'seg_mask_%i.npy'%im_ctr)
        np.save(filename, segmask)
    elif k == ord('n'):
        if im_ctr == len(input_imgs)-1:
            im_ctr = 0
        else:
            im_ctr += 1
        im_file = os.path.join(input_img_folder, input_imgs[im_ctr])
        img = cv2.imread(im_file)

