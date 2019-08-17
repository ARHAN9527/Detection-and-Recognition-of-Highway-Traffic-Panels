# -*- coding: utf-8 -*-
import time
from pathlib import Path

import cv2

class Video_processor:
    def __init__(self, video, save_path=(Path.cwd() / "new_frame")):
        # Capture video
        self.video = video
        self.capture = cv2.VideoCapture(self.video)
        if self.capture.isOpened() == False:
            print("Error opening file")
            return

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.spf = 1 / self.fps
        self.is_called = False
        self.running = False
        self.frame_counter = -1
        self.quit = [ord('q'), ord('Q')]
        self.rewind = [ord('b'), ord('B')]
        self.forward = [ord('f'), ord('F')]

        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.total_time = 0.0

    def stop(self):
        self.running = False

    def run(self):
        self.running = True

    def is_running(self):
        """Check status"""
        return self.running

    def callback_function(self, called_function):
        """Callback funtion"""
        self.called_function = called_function
        self.is_called = True

    def save_img(self, frame, file_name):
        file_name = str(self.save_path / file_name) + ".jpg"
        cv2.imwrite(file_name, frame)

    def get_video_fps(self):
        return self.fps

    def delay_correction(self):
        end_time = time.time()
        frame_time = end_time - self.start_time
        #print(frame_time)
        self.total_time = self.total_time + frame_time
        print("avg time=", self.total_time/(self.frame_counter+1))

        # Frame per second - duration
        delay_time = self.spf - (end_time - self.start_time)
        try:
            time.sleep(delay_time)
        except:
            pass

    def save_frames(self):
        print("\nStart save")

        while self.capture.isOpened():
            # Get frame, if no frames has been grabbed, ret == False
            ret, frame = self.capture.read()
            if ret == False:
                print("No next video frame")
                break

            self.frame_counter += 1

            # Save frame
            file_name = str(self.frame_counter).zfill(6)
            print(file_name)
            self.save_img(frame, file_name)
            #break

        # Close video
        self.capture.release()
        cv2.destroyAllWindows()

        print("End save\n")

    def show(self):
        """Display input and output frame"""
        print("\nStart capture")

        self.running = True

        while self.capture.isOpened():
            while self.is_running():
                self.start_time = time.time()
                # Pause
                key = cv2.waitKey(1)
                if  key != -1:
                    if key in self.rewind:
                        pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, pos-90)
                    elif key in self.forward:
                        pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, pos+90)
                    else:
                        self.stop()

                # Get frame, if no frames has been grabbed, ret == False
                ret, frame = self.capture.read()
                if ret == False:
                    print("No next video frame")
                    break

                self.frame_counter += 1
                # Display input frame
                #cv2.imshow('Input frame', frame)

                # Check callback function
                if self.is_called == True:
                    output_frame = self.called_function(frame)

                    # Display output frame
                    cv2.imshow('Output frame', output_frame)

                self.delay_correction()

            if ret == False:
                break

            # Resume or exit
            key = cv2.waitKey(1)
            if key != -1:
                if key in self.quit:
                    break
                self.run()

        # Close video
        self.capture.release()
        cv2.destroyAllWindows()

        print("End capture\n")

if __name__ == "__main__":
    VIDEO_NAME = r"demo_阿羅哈客運 3999 線路 國道一號 中山高速公路 高雄 - 台北 全程 路程景CUT_panel.mp4"
    SAVE_PATH = Path.cwd() / "night_frames3"

    #import tensorflow as tf
    import numpy as np
    from tensorflow.contrib import predictor
    from SSD.utils import preprocess_image, process_bboxes
    from SSD.visualization import bboxes_draw_on_img, bbox_draw_on_img

    def get_pred_fn(dir_path):
        export_dir = dir_path
        subdirs = [x for x in Path(export_dir).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1]) # saved_model/1556852589
        return predictor.from_saved_model(latest)

    detect_fn = get_pred_fn("./SSD/saved_model_mobilev2_ssd_width0125_38_extra05re") #saved_model_ssd_fpn_ratio75_size05 saved_model_fssd
    recognize_fn = get_pred_fn("./SSD/saved_model_recognition_model_mobilev2_width0125")

    def detect_predi(frame):
        #t1 = time.time()

        img_prepocessed = preprocess_image(frame, num=1)

        pred = detect_fn({'x': img_prepocessed})
        rclasses, rscores, rbboxes = process_bboxes(tflite=pred["tfoutput"]) # post process boxes

        #t2 = time.time()
        #print("Cost time = ", t2 - t1)

        return rclasses, rscores, rbboxes

    def recognize_predi(frame, img_rgb, rclasses, rscores, rbboxes):
        #t1 = time.time()
        shape = img_rgb.shape

        img = frame

        for i in range(rbboxes.shape[0]):
            bbox = rbboxes[i]
            # Draw bounding box...
            ymin = int(bbox[0] * shape[0])
            xmin = int(bbox[1] * shape[1])
            ymax = int(bbox[2] * shape[0])
            xmax = int(bbox[3] * shape[1])

            panel = img_rgb[ymin:ymax, xmin:xmax, :]
            panel_prepocessed = preprocess_image(panel, None, None, (128,128), num=1)

            pred = recognize_fn({'x': panel_prepocessed})
            predict_class = np.argmax(pred["tfoutput"][0])

            if pred["tfoutput"][0][predict_class] < 0.8:
                predict_class = 1
            else:
                img = bbox_draw_on_img(img, predict_class, rscores[i], bbox)
            #img = bboxes_draw_on_img(frame, pred["classes"], pred["scores"], pred["bboxes"])

        #t2 = time.time()
        #print("Cost time = ", t2 - t1)

        return img

    def predi(frame):
        #t1 = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rclasses, rscores, rbboxes = detect_predi(img_rgb)
        img = recognize_predi(frame, img_rgb, rclasses, rscores, rbboxes)

        #t2 = time.time()
        #print("Cost time = ", t2 - t1)

        return img

    t1 = time.time()

    processor = Video_processor(VIDEO_NAME, SAVE_PATH)
    processor.callback_function(predi)
    processor.show()
    #processor.save_frames()

    t2 = time.time()
    print("Cost time = ", t2 - t1)