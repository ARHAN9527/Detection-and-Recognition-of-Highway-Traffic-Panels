# -*- coding: utf-8 -*-
import random
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import preprocess_image

class Read_txt:
    def __init__(self, dir_path, txt_name, size):
        self.dir_path = dir_path
        self.txt_name = txt_name
        self.size = size
        self.labels = []

    def get_labels(self):
        return self.labels

    def read(self):
        '''
        Label with file name, bboxes and classes
        Return:
        a list with dicts,
        name: image file path
        bbox: ground truth with 3d [1, num_boxes, 4] (ymin,xmin,ymax,xmax)
        class: label of boxes with 3d [1, num_boxes]
        '''
        with open(self.txt_name,"r") as fp:
            all_lines = fp.readlines()

        img_w = self.size[0]-1
        img_h = self.size[1]-1

        for line in all_lines:
            line = line[:-1].split(",")
            name = line[0]
            num = int(line[1])
            vertex = []
            rec = []
            bbox = []
            cla_list = []

            for i in range(num):
                index = 5 * i
                '''x1 = int(line[2+index])
                y1 = int(line[3+index])
                x2 = int(line[4+index])
                y2 = int(line[5+index])'''

                #x1,y1,x2,y2 = map(int, [line[2+index],line[3+index],line[4+index],line[5+index]])
                x1,y1,x2,y2,cla = map(lambda x: int(line[x+index]), [2,3,4,5,6])

                '''x1 = x1 if x1 < img_w else (img_w-1)
                y1 = y1 if y1 < img_h else (img_h-1)
                x2 = x2 if x2 < img_w else (img_w-1)
                y2 = y2 if y2 < img_h else (img_h-1)'''

                x1,y1,x2,y2 = map(lambda x, y: min(x, y), [x1,y1,x2,y2], [img_w,img_h,img_w,img_h])

                #cx = (x2 + x1) // 2
                #cy = (y2 + y1) // 2
                #w = x2 - x1
                #h = y2 - y1

                #vertex.append({"top_left": (x1, y1), "bottom_right": (x2, y2)})
                #rec.append({"cx": cx, "cy": cy, "w": w, "h": h})
                y1 = np.maximum(y1/img_h, 0)
                x1 = np.maximum(x1/img_w, 0)
                y2 = np.minimum(y2/img_h, 1)
                x2 = np.minimum(x2/img_w, 1)
                
                bbox.append([y1, x1, y2, x2])
                cla_list.append(cla)

            self.labels.append({"name": str(self.dir_path/name), "bbox": [bbox], "class": [cla_list]})

        return self.labels

    def copy_img(self, src_path, dst_path):
        import shutil

        for i, img in enumerate(self.labels):
            try:
                shutil.copy(str(src_path / img["name"]), str(dst_path))
                print(i, " ", img["name"], "done")
            except shutil.Error:
                print(i, " ", img["name"], " error!")

    def move_img(self, src_path, dst_path):
        import shutil

        for i, img in enumerate(self.labels):
            try:
                shutil.move(str(src_path / img["name"]), str(dst_path))
                print(i, " ", img["name"], "done")
            except FileNotFoundError:
                print(i, " ", img["name"], " is missing")
            except shutil.Error:
                print(i, " ", img["name"], " already exists!")

    def create_binary_img(self, src_path, dst_path, show = False):
        for file in self.labels:
            img = cv2.imread(str(src_path / file["name"]))
            if img is None:
                continue

            img_binary = np.zeros((img.shape[0], img.shape[1]), np.uint8)

            for j in range(file["num"]):
                top_left = file["vertex"][j]["top_left"]
                bottom_right = file["vertex"][j]["bottom_right"]
                cv2.rectangle(img, top_left, bottom_right, (0, 0 , 255), 2)
                cv2.rectangle(img_binary, top_left, bottom_right, (255, 255 , 255), -1)

            cv2.imwrite(str(dst_path / file["name"]), img_binary)
            print(file["name"], "done")

            cv2.imshow("img", img)
            cv2.imshow("img_binary", img_binary)

            key = cv2.waitKey(1)
            if key == ord('Q'):
                cv2.destroyAllWindows()
                break

    def create_panels(self, src_path, dst_path, labels):
        for file in labels:
            img = cv2.imread(file["name"])
            if img is None:
                continue
            print("load: ", file["name"])    
                
            for i in range(len(file["bbox"][0])):
                ymin = int(file["bbox"][0][i][0] * img.shape[0])
                xmin = int(file["bbox"][0][i][1] * img.shape[1])
                ymax = int(file["bbox"][0][i][2] * img.shape[0])
                xmax = int(file["bbox"][0][i][3] * img.shape[1])
                
                panel = img[ymin:ymax, xmin:xmax, :]               
                cv2.imwrite(str(dst_path / (file["name"].split("\\")[-1].split(".")[0]+"_"+str(i)+".jpg")), panel)
                print(file["name"].split("\\")[-1].split(".")[0]+"_"+str(i)+".jpg", "done")
                #cv2.imshow("panel"+str(j), panel)    
                #fig = plt.figure(figsize=(10,10))   
                #plt.imshow(panel[:,:,::-1])
                #plt.show()

            #break

            '''key = cv2.waitKey(1)
            if key == ord('Q'):
                cv2.destroyAllWindows()
                break'''

    def show_img(self, src_path):
        for file in self.labels:
            img = cv2.imread(str(src_path / file["name"]))
            if img is None:
                continue

            for j in range(file["num"]):
                #cv2.rectangle(img, file["vertex"][j][0], file["vertex"][j][1], (0, 0 , 255), 2)
                cx = file["rec"][j]["cx"]
                cy = file["rec"][j]["cy"]
                w = file["rec"][j]["w"]
                h = file["rec"][j]["h"]
                x1 = (cx-(w//2))
                y1 = (cy-(h//2))
                x2 = (cx+(w//2))
                y2 = (cy+(h//2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0 , 255), 2)

            cv2.imshow("img", img)
            #print(file)

            key = cv2.waitKey(0)
            if key == ord('Q'):
                cv2.destroyAllWindows()
                break


def save_dataset(labels,
                 image_dir_path,
                 file_image_size = 10000,
                 save_path = None):

    print("\nStart save file")

    if save_path is None:
        save_path = Path.cwd() / "npz"

    dir_name = image_dir_path.stem
    targets = labels[:2048]
    random.shuffle(targets)
    img_list = []
    bbox_list = []
    label_list = []
    counter = -1

    for i, target in enumerate(targets):
        # Read image file
        img = cv2.imread(str(image_dir_path / target["name"]))
        if img is None:
            continue
        print(i, target["name"])

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        #bbox = np.array(target['bbox'], np.float32)
        bbox = target['bbox']
        label = target['class']

        if len(label[0]) < 4:
            for i in range(4-len(label[0])):
                label[0].append(0)
                bbox[0].append([0,0,0,0])

        img_list.append(np.expand_dims(img_rgb, axis=0))
        bbox_list.append(np.asarray(bbox, np.float32))
        label_list.append(np.asarray(label, np.int64))
        counter += 1

        left = len(targets) - (counter + 1)
        # Save file
        if ((counter+1) % file_image_size) == 0 or left == 0:
            img_array = np.concatenate(img_list)
            bbox_array = np.concatenate(bbox_list)
            label_array = np.concatenate(label_list)

            np.savez_compressed(str(save_path / (dir_name + "_" + str(counter // file_image_size).zfill(4))),
                                image = img_array,
                                bbox = bbox_array,
                                label = label_array)


            print("save_name: ", str(save_path / (dir_name + "_" + str(counter // file_image_size).zfill(4))))
            print("img_array.shape: ", img_array.shape)
            print("bbox_array.shape: ", bbox_array.shape)
            print("label_array.shape: ", label_array.shape)

            img_list = []
            bbox_list = []
            label_list = []

    # Last time
    if left != 0:
        img_array = np.concatenate(img_list)
        bbox_array = np.concatenate(bbox_list)
        label_array = np.concatenate(label_list)

        np.savez_compressed(str(save_path / (dir_name + "_" + str(counter // file_image_size).zfill(4))),
                            image = img_array,
                            bbox = bbox_array,
                            label = label_array)

        print("save_name: ", str(save_path / (dir_name + "_" + str(counter // file_image_size).zfill(4))))
        print("img_array.shape: ", img_array.shape)
        print("bbox_array.shape: ", bbox_array.shape)
        print("label_array.shape: ", label_array.shape)

    print("End save file\n")

def load_dataset(file_path, begin, end):
    print("\nStart load file")

    img_list = []
    bbox_list = []
    label_list = []

    npz_files = list(Path(file_path).glob("*.npz"))
    npz_files.sort()

    for file_name in npz_files[begin:end]:
        data = np.load(str(file_name))
        img_list.append(data["image"])
        bbox_list.append(data["bbox"])
        label_list.append(data["label"])

        print("load:", str(file_name))

    img_array = np.concatenate(img_list).astype(np.float32)
    bbox_array = (np.concatenate(bbox_list)).astype(np.float32)
    label_array = (np.concatenate(label_list)).astype(np.int64)

    dataset = {"image": img_array,
               "bbox": bbox_array,
               "class": label_array,}

    print("End load file\n")

    return dataset

def load_npz(file_path):
    print("\nStart load file")

    img_list = []
    bbox_list = []
    label_list = []

    data = np.load(str(file_path))
    img_list.append(data["image"])
    bbox_list.append(data["bbox"])
    label_list.append(data["label"])

    print("load:", str(file_path))

    img_array = np.concatenate(img_list).astype(np.float32)
    bbox_array = (np.concatenate(bbox_list)).astype(np.float32)
    label_array = (np.concatenate(label_list)).astype(np.int64)

    dataset = {"image": img_array,
               "bbox": bbox_array,
               "class": label_array,}

    print("End load file\n")

    return dataset

def save_dataset2(labels,
                  image_dir_path,
                  file_image_size = 10000,
                  save_path = None,
                  num=0):
    print("\nStart save file")

    if save_path is None:
        save_path = Path.cwd() / "npz_day_night_nostan"

    dir_name = image_dir_path.stem
    targets = labels   

    def save_npz(img_list, label_list, bbox_list, save_path, dir_name, counter, file_image_size, loop):        
        img_array = np.concatenate(img_list)
        label_array = np.concatenate(label_list)
        bbox_array = np.concatenate(bbox_list)

        np.savez_compressed(str(save_path / (dir_name + "_" + str(loop) + "_" + str(counter // file_image_size).zfill(4))),
                            image = img_array,
                            label = label_array,
                            bbox = bbox_array)


        print("save_name: ", str(save_path / (dir_name + "_" + str(loop) + "_" + str(counter // file_image_size).zfill(4))))
        print("img_array.shape: ", img_array.shape)
        print("label_array.shape: ", label_array.shape)
        print("bbox_array.shape: ", bbox_array.shape)

    for loop in range(2):
        random.shuffle(targets)
        counter = -1
        img_list = []
        label_list = []
        bbox_list = []
        for idx, target in enumerate(targets):
            img = cv2.imread(target["name"])
            if img is None:
                continue
            print(idx, "load: ", target["name"])    
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #bboxes = preprocess_label(target['bbox'], img.shape)
            bboxes = np.array(target['bbox'], np.float32)
            labels = target['class']
            
            img_prepocessed, label_prepocessed, \
                box_prepocessed,cropped_image_with_box = preprocess_image(img_rgb, 
                                                                          labels,
                                                                          bboxes,
                                                                          out_shape=(300,300),
                                                                          min_object_covered=0.9,
                                                                          aspect_ratio_range=(1.0, 1.0),
                                                                          num=num)
            if label_prepocessed.shape[1] < 4:
                label_prepocessed = label_prepocessed.tolist()
                box_prepocessed = box_prepocessed.tolist()
                for i in range(4-len(label_prepocessed[0])):            
                    label_prepocessed[0].append(0)
                    box_prepocessed[0].append([0,0,0,0])

            label_prepocessed = np.asarray(label_prepocessed, np.int64)
            box_prepocessed = np.asarray(box_prepocessed, np.float32)

            if label_prepocessed.size == 0:
                continue

            img_list.append(img_prepocessed)
            label_list.append(label_prepocessed)
            bbox_list.append(box_prepocessed)

            counter += 1

            left = len(targets) - (counter + 1)
            # Save file
            if ((counter+1) % file_image_size) == 0 or left == 0:
                save_npz(img_list, label_list, bbox_list, save_path, dir_name, counter, file_image_size, loop)
                img_list = []
                label_list = []
                bbox_list = []

        # Last time
        if left != 0:
            save_npz(img_list, label_list, bbox_list, save_path, dir_name, counter, file_image_size, loop)

    print("End save file\n")

if __name__ == "__main__":
    import time

    #dir_path = Path.cwd().parent / "night_frames2"
    #labels1 = Read_txt(dir_path, str( dir_path / "night_frames2_gt_cls.txt"), (1280, 720)).read() # 1276

    #dir_path = Path.cwd().parent / "day_frames"
    #labels2 = Read_txt(dir_path, str( dir_path / "day_frames_gt.txt"), (1280, 720)).read() # 6232
    
    dir_path = Path.cwd().parent / "night_frames"
    a = Read_txt(dir_path, str( dir_path / "night_frames_gt.txt"), (1280, 720))
    labels3 = a.read() # 2697
    
    #random.shuffle(labels1)
    #random.shuffle(labels2)
    #random.shuffle(labels3)
    
    
    #labels = labels1[:1024] + labels2[:2048] + labels3[:1024] # 4096
    labels = labels3 # 
    print('labels len= ', len(labels))

    src_path = dir_path
    dst_path = Path.cwd().parent / "night_frames_panel"

    t1 = time.time()

    #print(labels[50])

    #a.copy_img(src_path, dst_path)
    #a.show_img(src_path)

    #a.create_binary_img(src_path, dst_path)
    #a.move_img(src_path, dst_path)
    a.create_panels(src_path, dst_path, labels)
    #save_dataset(labels, src_path, 32)
    #save_dataset2(labels, src_path, 32)
    #dataset = load_dataset(Path.cwd())
    '''print(dataset["class"].shape)
    print(dataset["bbox"].shape)
    print(dataset["image"].shape)'''

    t2 = time.time()
    print("\nCost time = ", t2 - t1)