import json
import os
import math
import re
import cv2
import numpy as np
import sys
from colorthief import ColorThief


def get_dominant_colours(images, outfile):
    output = {'colors': []}
    avg_r = 0
    avg_b = 0
    avg_g = 0
    count = 0
    for i, img in enumerate(images):
        avg_color_per_row = np.average(img['img'], axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        o = {'file_name': img['file_name'], 'color': list(avg_color)}
        output['colors'].append(o)
        sys.stdout.write(f"\rimages analyzed: {i+1}/{len(images)}")
        sys.stdout.flush()

    hero_rgb = (avg_r, avg_g, avg_b)
    regex = re.compile(r"\D*")
    output = sorted(output['colors'], key=lambda x: int(
        re.sub(regex, '', x['file_name'])))
    return output


def get_dominant_color():
    c_array = []
    output = {'colors': []}
    with open('colours/hero_colours.json', 'w') as outfile:
        with open('json_files/hero_ids.json', 'r') as f:
            data = json.load(f)
            for item in data['heroes']:
                hero_name = item['name']
                avg_r = 0
                avg_b = 0
                avg_g = 0
                count = 0
                for i, filename in enumerate(os.listdir(f'colours/ability_images/{hero_name}')):
                    try:
                        c_t = ColorThief(
                            f"colours/ability_images/{hero_name}/{filename}")
                        d_c = c_t.get_color(quality=1)
                        r = d_c[0]
                        g = d_c[1]
                        b = d_c[2]
                        avg_r += r
                        avg_g += g
                        avg_b += b
                        count += 1
                        # print(hero_name)
                    except Exception as e:
                        print(filename, hero_name, e)
                # print([sum(i) for i in avg_color])
                try:
                    avg_r /= count
                    avg_g /= count
                    avg_b /= count
                    rgb = (avg_r, avg_g, avg_b)
                    o = {'hero': hero_name, 'color': rgb}
                    output['colors'].append(o)
                except Exception as e:
                    print(hero_name, e)
        json.dump(output, outfile, indent=4)


def nearest_colour(subjects, query):
    # numpy.sqrt((r1 - r0)**2 + (g1 - g0)**2 + (b1 - b0)**2.)
    best = {'img': '', 'euc': 10000}
    for x in subjects:
        r1 = x['color'][0]
        g1 = x['color'][1]
        b1 = x['color'][2]
        b0 = query[0]
        g0 = query[1]
        r0 = query[2]
        euc = math.sqrt((b1 - b0)**2 + (g1 - g0)**2 + (r1 - r0)**2)
        if euc <= best['euc']:
            best['euc'] = euc
            best['img'] = x['ability']
    return best['img']
    # return min( subjects, key = lambda subject: sum( (s - q) ** 2 for s, q in zip( subject, query ) ) )


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def nearest_img(img_lst,x):
    debug = True
    m_imgs = []
    regex = re.compile(r"\D*")
    print('x', x)
    for i,color in enumerate(img_lst):
        target_color = color['color']
        with open('mosaic_colours.json', 'r') as f:
            c_data = json.load(f)
            mosaic_colors = c_data['colors']
            m_imgs.append(nearest_colour(mosaic_colors, target_color))
        sys.stdout.write(f"\rnearest img found: {i+1}/{len(img_lst)}")
        sys.stdout.flush()
    return list(chunks(m_imgs, x))


# get_dominant_colours('images/singles','target_colours')
# get_dominant_colours('mosaic_imgs','mosaic_colours')


def combine_image(x, y):
    final_img = []
    for i in range(y):
        directory = os.listdir(f'images/slice_{i}')
        for j in range(x):
            if j == 0:
                vertical_slice = cv2.imread(f'images/slice_{i}/{directory[j]}')
            if j < len(directory)-1:
                next_img = cv2.imread(f"images/slice_{i}/{directory[j+1]}")
                vertical_slice = cv2.hconcat(
                    [vertical_slice, next_img])
        cv2.imwrite(f"fin/slice_{i}.png", vertical_slice)
    for i in range(y-1):
        if len(final_img) == 0:
            final_img = cv2.imread(f'fin/slice_{i}.png')
        next_img = cv2.imread(f'fin/slice_{i+1}.png')
        final_img = cv2.vconcat([final_img, next_img])
    cv2.imwrite(f"fin/final_img.png", final_img)


if __name__ == '__main__':

    pass
