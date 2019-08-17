# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import random
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm


DRAW_FONT = ImageFont.truetype("./NotoSansCJKtc-hinted/NotoSansCJKtc-Bold.otf", 15) #encoding="utf-8"
DRAW_FILL_COLOR = (0,128,255)

# class names
CLASSES = ["Panel",
            "頭屋 造橋 右線",
            "國道三號",
            "頭份",
            "頭屋 造橋 箭頭右",
            "頭份 14 國道三號 25",
            "頭份 6 國道三號 17",
            "旅行時間標誌板 頭份 新竹 中壢",
            "頭份 三灣 出口2公里",
            "頭份 三灣 右線",
            "新竹",
            "頭份 三灣 箭頭右",
            "國道三號 9 新竹 13",
            "竹南 竹東 2000m",
            "竹南 竹東 1500m",
            "台北 竹東 右線",
            "竹北",
            "竹南",
            "台北 竹東 箭頭右",
            "後龍 竹南 右線",
            "後龍 竹南 箭頭右",
            "新竹科學園區 右線",
            "湖口",
            "新竹 竹東 出口1.5公里",
            "新竹科學園區 箭頭右",
            "新竹 竹東 右線",
            "楊梅",
            "新竹 竹東 箭頭右",
            "竹北 芎林 右線",
            "竹北 芎林 箭頭右",
            "中壢 林口 約 分",
            "新豐 湖口 新竹工業區 出口2公里",
            "新豐 湖口 新竹工業區 右線",
            "中壢",
            "新豐 湖口 新竹工業區 箭頭右",
            "楊梅 12 中壢 19",
            "楊梅 中壢 2500m",
            "楊梅 中壢 1000m",
            "66快速道路",
            "中壢 桃園機場",
            "楊梅 埔心 右線",
            ]
'''
CLASSES = ["panel", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
'''
# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

def draw_text(img, str, pos):
    pilimg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilimg)
    draw.text(pos, str, DRAW_FILL_COLOR, font=DRAW_FONT)
    return np.array(pilimg)

def bbox_draw_on_img(img, classes, score, bbox, colors=(0,128,255), thickness=1):
    shape = img.shape
    #color = colors[classes[i]]
    color = colors
    # Draw bounding box...
    p1 = (int(bbox[1] * shape[1]), int(bbox[0] * shape[0]))
    p2 = (int(bbox[3] * shape[1]), int(bbox[2] * shape[0]))
    cv2.rectangle(img, p1, p2, color, thickness)
    # Draw text...
    s = '%s' % (CLASSES[classes-1])
    #s = '%s.%.3f' % (CLASSES[classes[i]-1], score)
    p1 = (p1[0], p2[1])
    #p1 = (p1[0], p1[1]-5)
    #cv2.putText(img, s, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    img = draw_text(img, s, p1)

    return img

def bboxes_draw_on_img(img, classes, scores, bboxes, colors=(0,128,255), thickness=5):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        #color = colors[classes[i]]
        color = colors
        # Draw bounding box...
        p1 = (int(bbox[1] * shape[1]), int(bbox[0] * shape[0]))
        p2 = (int(bbox[3] * shape[1]), int(bbox[2] * shape[0]))
        cv2.rectangle(img, p1, p2, color, thickness)
        # Draw text...
        s = '%s' % (CLASSES[classes[i]-1])
        #s = '%s.%.3f' % (CLASSES[classes[i]-1], scores[i])
        p1 = (p1[0], p2[1])
        #p1 = (p1[0], p1[1]-5)
        #cv2.putText(img, s, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        img = draw_text(img, s, p1)

    return img


# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5, show_class_name=True):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = 'blue'#(random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = CLASSES[cls_id-1] if show_class_name else str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s},{:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    #plt.axis('off')
    #plt.savefig("test_img/filename.png")
    plt.show()
