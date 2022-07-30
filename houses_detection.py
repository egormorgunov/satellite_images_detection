import operator
import os
import numpy as np
from skimage import io, draw, transform, exposure, color


'''Функция считывания изображений'''
def read_images(path):
    images_list = []
    for image_name in os.listdir(path):
        img = io.imread(path + '/' + image_name)
        img_gray = color.rgb2gray(img)
        img_contrast = exposure.adjust_sigmoid(img_gray, cutoff=0.7, gain=25, inv=False)
        images_list.append((img, img_contrast))
    return images_list


'''Функция сохранения изображений'''
def save_image(path, no, img):
    return io.imsave(path + '_' + str(no + 1) + '.jpg', img)


'''Функция вычисления признаков Хаара определенной формы'''
def get_haar_params(dark_w, bright_w, h):
    return [[(0, 0), (h, bright_w)], [(0, bright_w), (h, dark_w + bright_w)],
            [(0, bright_w + dark_w), (h, 4 * bright_w + dark_w)]]


'''Функция детектирования объектов'''
def detect_object(hc_img, w_threshold):
    res_coords = []
    int_img = transform.integral_image(hc_img)
    [area1, area2, area3] = get_haar_params(dark_w=1, bright_w=2, h=15)
    square1 = area1[1][0] * (area1[1][1] - area1[0][1])
    square2 = area2[1][0] * (area2[1][1] - area2[0][1])
    square3 = area3[1][0] * (area3[1][1] - area3[0][1])
    for angle in range(0, 360, 10):
        rotated_haar = transform.rotate(np.zeros(area3[1], dtype=np.uint8), angle, resize=True, preserve_range=False)
        window_w, window_h = rotated_haar.shape
    for x in range(0, hc_img.shape[0] - 2 * window_w - 1, 5):
        for y in range(0, hc_img.shape[1] - 2 * window_h - 1, 5):
            mean_bright1 = transform.integrate(int_img, (x, y), (x + area1[1][0], y + area1[1][1])) / square1
            mean_dark = transform.integrate(int_img, (x + area2[0][0], y + area2[0][1]),
                                            (x + area2[1][0], y + area2[1][1])) / square2
            mean_bright2 = transform.integrate(int_img, (x + area3[0][0], y + area3[0][1]),
                                               (x + area3[1][0], y + area3[1][1])) / square3
            mean_bright = (mean_bright1 + mean_bright2) / 2
            if mean_bright - mean_dark > w_threshold:
                res_coords.append((x, y, (window_w, window_h)))
    res_coords = list(set(res_coords))
    cleaned_coords = remove_close_hits(res_coords)
    return cleaned_coords


'''Функция удаления повторяющихся координат'''
def remove_close_hits(res_coords):
    new_coords = sorted(res_coords, key=operator.itemgetter(0))
    i = 1
    n = len(new_coords)
    while i < n:
        x = new_coords[i][0]
        y = new_coords[i][1]
        x1 = new_coords[i - 1][0]
        y1 = new_coords[i - 1][1]
        difx = abs(x - x1)
        dify = abs(y - y1)
        if (difx + dify) <= 240:
            del new_coords[i]
            n -= 1
        else:
            i += 1
    return new_coords


'''Сохранение изображения, наложение рамки на найденный объект'''
def save_result(img, coords, im_no):
    for x, y, window_shape in coords:
        coord = x, y
        if coord != (-1, -1):
            window_shape = (40, 80)
            rr, cc = draw.rectangle_perimeter(coord, extent=window_shape, shape=img.shape)
            img[rr - 5, cc - 5] = (0, 255, 0)
        save_image("houses_detected/img", im_no, img)


dataset = read_images('houses')
dark_and_bright_mean_value = 0.30
for image_number, image in enumerate(dataset):
    print('Идет обработка изображения №', image_number + 1)
    coordinates = detect_object(image[1], dark_and_bright_mean_value)
    save_result(image[0], coordinates, image_number)
