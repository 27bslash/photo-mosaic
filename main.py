import os
import re
import shutil
import numpy as np
import cv2
import json
import sys
from get_dominant_colour import nearest_img, get_dominant_colours

# enter image name here
target_img = cv2.imread("aa.jpg")

scale = 2
width = target_img.shape[1]
height = target_img.shape[0]
target_img = cv2.resize(target_img, (int(width*scale), int(height*scale)))

cv2.imwrite('target.jpg', target_img)


def chunk_image(x, y, scale):
    img_shape = target_img.shape
    w = int(img_shape[1] / x)
    h = int(img_shape[0] / y)
    if w < 1:
        w = 1
        x = img_shape[1]
    if h < 1:
        h = 1
        y = img_shape[0]
    tile_size = (w, h)
    offset = (w, h)

    shutil.rmtree('images')
    shutil.rmtree('solid')
    os.mkdir('images')
    os.mkdir('solid')
    os.mkdir('images/singles')

    count = 0
    imgs = []
    print(img_shape, offset)
    print(f'chunking image into: {x}x{y}')
    for i in range(y):
        for j in range(x):
            cropped_img = target_img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]),
                                     offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
            # cv2.imwrite(f"images/singles/image_{count}.jpg", cropped_img)
            imgs.append({"file_name": f"image_{count}", "img": cropped_img})
            count += 1
            # cv2.imwrite(f"images/slice_{i}/img_{j}.png", cropped_img)
    colors = get_dominant_colours(imgs, 'target_colours')

    # get_dominant_colours('mosaic_imgs','mosaic_colours')
    combine(colors, x, y)


def combine(colors, x, y):
    h_slices = []
    final_img = []
    lst = nearest_img(colors, x)
    resize_images(lst, x, y)
    print('Combining images...')
    for i in range(len(lst)):
        h_slice = 0
        for j in range(len(lst[i])):
            if j == 0:
                # h_slice = lst[i][j]['image']
                h_slice = cv2.imread(f"resized/{lst[i][j]}")
                continue
                # h_slice = resize_image(h_slice, x, y)
            # next_img = lst[i][j]['image']
            try:
                next_img = cv2.imread(f"resized/{lst[i][j]}")
                # next_img = resize_image(next_img, x, y)
                h_slice = cv2.hconcat(
                    [h_slice, next_img])
            except Exception as e:
                pass
        if i == 0:
            final_img = h_slice
            continue
        else:
            next_img = h_slice
            final_img = cv2.vconcat([final_img, next_img])
        sys.stdout.write(f"\rSlices combined: {i+1}/{len(lst)}")
        sys.stdout.flush()
    cv2.imwrite('final_image.jpg', final_img)


def resize_image(img, x, y):
    img_shape = target_img.shape
    width = img_shape[1]/x
    height = img_shape[0]/y
    resized = cv2.resize(img, (width, height))
    return resized


def resize_images(lsts, x, y):
    img_shape = target_img.shape
    w = int(img_shape[1]/x)
    h = int(img_shape[0]/y)
    img_set = set()
    shutil.rmtree('resized')
    os.mkdir('resized')
    print('\nresizing images...', len(lsts))
    for lst in lsts:
        for image in lst:
            if image not in img_set:
                resized = cv2.resize(cv2.imread(
                    f"mosaic_imgs/{image}"), dsize=(w, h))
                cv2.imwrite(f"resized/{image}", resized)
                img_set.add(image)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def solid_colour(x):
    lst = []
    with open('target_colours.json', 'r') as f:
        data = json.load(f)
        for item in data['colors']:
            image = np.zeros((300, 300, 3), np.uint8)
            # Fill image with red color(set each pixel to red)
            new_color = (item['color'][2], item['color'][1], item['color'][0])
            image[:] = new_color

            cv2.imwrite(f'solid/{item["ability"]}', image)
            # lst.append(image)
    dire = os.listdir('solid')
    regex = re.compile(r"\D*")
    # dire = [int(re.sub(regex, '', x)) for x in dire]
    dire = sorted(dire, key=lambda x: int(re.sub(regex, '', x)))
    # for filepath in glob.iglob('images/singles/*'):
    #     print(filepath)
    # for i in range(len(os.listdir('images/singles'))):
    #     print(os.listdir('images/singles')[i])
    for file in dire:
        lst.append(
            {"file_name": file, "image": cv2.imread(f"solid/{file}")})
    # print(lst)
    return list(chunks(lst, x))


if __name__ == "__main__":
    chunk_image(500, 500,1)
    pass
