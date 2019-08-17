from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ssd_layers import *

class Recognition_model:
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.recog_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                      model_dir=model_path)

        self.predictor = tf.contrib.predictor.from_estimator(self.recog_estimator, Recognition_model.serving_input_receiver_fn)

    def _backbone(self, x, is_training):
        exp = 6
        width_multiplier = 0.125
        
        # Input Layer
        net = tf.reshape(x, [-1, 128, 128, 3], name="input_tensor")
        
        # mobilev2
        # layer1: input[batch,300,300,3], output[batch,150,150,32]
        net = conv_bn_relu(net, 32, 3, 2, is_training, scope="conv1_1") # size/2 150
        self.end_points["layer1"] = net
        # layer2: input[batch,150,150,32], output[batch,150,150,16]
        net = res_block(net, 1, 16*width_multiplier, 1, is_training, shortcut=False, scope='res2_1') # 不进行残差合并 f(x)
        self.end_points["layer2"] = net
        # layer3: input[batch,150,150,16], output[batch,75,75,24]
        net = res_block(net, exp, 24*width_multiplier, 2, is_training, scope='res3_1') # size/4 75
        net = res_block(net, exp, 24*width_multiplier, 1, is_training, scope='res3_2')
        self.end_points["layer3"] = net
        # layer4: input[batch,75,75,24], output[batch,38,38,32]
        net = res_block(pad2d(net, 1), exp, 32*width_multiplier, 2, is_training, scope='res4_1', padding="valid") # size/8 38
        net = res_block(net, exp, 32*width_multiplier, 1, is_training, scope='res4_2')
        net = res_block(net, exp, 32*width_multiplier, 1, is_training, scope='res4_3')
        self.end_points["layer4"] = net
        # layer5: input[batch,38,38,32], output[batch,19,19,64]
        net = res_block(net, exp, 64*width_multiplier, 2, is_training, scope='res5_1') # size/16 19
        net = res_block(net, exp, 64*width_multiplier, 1, is_training, scope='res5_2')
        net = res_block(net, exp, 64*width_multiplier, 1, is_training, scope='res5_3')
        net = res_block(net, exp, 64*width_multiplier, 1, is_training, scope='res5_4')
        self.end_points["layer5"] = net
        # layer6: input[batch,19,19,64], output[batch,19,19,96]
        net = res_block(net, exp, 96*width_multiplier, 1, is_training, scope='res6_1')
        net = res_block(net, exp, 96*width_multiplier, 1, is_training, scope='res6_2')
        net = res_block(net, exp, 96*width_multiplier, 1, is_training, scope='res6_3')
        self.end_points["layer6"] = net
        # layer7: input[batch,19,19,96], output[batch,10,10,160]
        net = res_block(pad2d(net, 1), exp, 160*width_multiplier, 2, is_training, scope='res7_1', padding="valid") # size/32 10
        net = res_block(net, exp, 160*width_multiplier, 1, is_training, scope='res7_2')
        net = res_block(net, exp, 160*width_multiplier, 1, is_training, scope='res7_3')
        self.end_points["layer7"] = net
        # layer8: input[batch,10,10,160], output[batch,10,10,320]
        net = res_block(net, exp, 320*width_multiplier, 1, is_training, shortcut=False, scope='res8_1') # 不进行残差合并 f(x)
        self.end_points["layer8"] = net        
        ####### 9. 1个 1*1点卷积块 = 1*1 PW点卷积 +  BN + RELU6 1280个通道输出 ################
        net = conv_bn_relu(net, 512, 1, 1, is_training, scope="layer9_1") 
        self.end_points["layer9"] = net
        ####### 10. 全局均值 池化 average_pooling2d #########################################
        net = avg_pool2d(net, scope="avg_10_1")
        self.end_points["layer10"] = net
        ####### 11. 1*1 PW点卷积 后 展开成1维 ###############################################     
        net = conv2d(net, 42, 1, activation=None, scope="conv11_1") # activation=None
        net = tf.squeeze(net, [1, 2], name="squeeze11_2")
        self.end_points["layer11"] = net
        
        print(self.end_points)
        
        return net
    
    def _built_net(self, features, is_training):
        """
        Build SSD model with backbone and extra layer.

        Arguments:
        features: model input. 4D tensor: [batch, size, size, 3];
        is_training: true with training mode;

        Return:
        logits: list of classes with shape [batch, size*size, n_anchors, num_classes];
        locations: list of locations with shape [batch, size*size, n_anchors, 4];
        """
        self.end_points = {}  # record the detection layers output

        logits = self._backbone(features["x"], is_training)

        return logits
    
    def model_fn(self, features, labels, mode):
        """Model function for Recognitiont."""
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        logits = self._built_net(features, is_training)
        tfoutput = tf.nn.softmax(logits, name="softmax_tensor")

        predictions_tfoutput = {
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "tfoutput": tfoutput
        }
        export_outputs = {"predictions": tf.estimator.export.PredictOutput(predictions_tfoutput)}
        export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs['predictions']

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              predictions=predictions_tfoutput, 
                                              export_outputs=export_outputs)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels['class'],
                logits=logits)        
        
        ''' ##########################################################
        loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits)
        '''
        
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(1e-3, tf.train.get_global_step(), 40000, 0.1, staircase = True)
            optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss=loss,
                                          global_step=tf.train.get_global_step())
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#############################################
            train_op = tf.group([train_op, update_ops])#########################################################
            
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)
        
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                           predictions=predictions["classes"])}
        
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    def train(self, data, dataset_path = "./frame/panels/npz/train", steps=20000):
        '''
        # Train data
        dataset = Dataset_process().load_dataset(dataset_path)

        train_images = (dataset["image"])
        train_labels = (dataset["label"])

        if dataset != None:
            np.set_printoptions(threshold=np.nan)
            print("image_shape = ", train_images.shape)
            print("image_type = ", train_images.dtype)
            print("label_shape = ",train_labels.shape)
            print("label_type = ", train_labels.dtype)
            #cv2.imshow("photo", train_labels[0])
            #cv2.waitKey (0)
        '''
        
        train_images = data["img"]
        train_labels = {'class': np.asarray(data["class"], np.int64)}

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
                           tensors=tensors_to_log,
                           every_n_iter=100)

        # Train the model
        print("----Start train----")

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                             x={"x": train_images},
                             y=train_labels,
                             batch_size=data["batch"],
                             num_epochs=None,
                             shuffle=True)

        self.recog_estimator.train(input_fn=train_input_fn,
                                   steps=steps,
                                   hooks=[logging_hook])

        print("----End train----")

    def eval(self, dataset_path = "./frame/panels/npz/eval"):
        '''
        # Eval data
        dataset = Dataset_process().load_dataset(dataset_path)

        eval_images = (dataset["image"])
        eval_labels = (dataset["label"])

        if dataset != None:
            np.set_printoptions(threshold=np.nan)
            print("image_shape = ", eval_images.shape)
            print("image_type = ", eval_images.dtype)
            print("label_shape = ",eval_labels.shape)
            print("label_type = ", eval_labels.dtype)
            #cv2.imshow("photo", train_labels[0])
            #cv2.waitKey (0)
        '''
        eval_images = self.eval_data
        eval_labels = self.eval_labels

        # Train the model
        print("----Start eval----")

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                            x={"x": eval_images},
                            y=eval_labels,
                            num_epochs=1,
                            shuffle=False)

        eval_results = self.recog_estimator.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        print("----End eval----")

    def serving_input_receiver_fn():
        x = tf.placeholder(dtype=tf.float32, shape=[1, 128, 128, 3], name='x')
        #x = tf.placeholder(dtype=tf.float32, shape=[1, 784], name='x')
        features = receiver_tensors = {'x': x}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    def save_model(self, dir_path):        
        self.recog_estimator.export_savedmodel(dir_path, Recognition_model.serving_input_receiver_fn)
    
    def test(self, data=None, dataset_path = "./frame/panels/npz/train"):
        return self.predictor(data)
        #import cv2
        '''
        # Test data
        dataset = Dataset_process().load_dataset(dataset_path)

        test_images = (dataset["image"])
        test_labels = (dataset["label"])

        if dataset != None:
            np.set_printoptions(threshold=np.nan)
            #print("image[0] = \n", test_images[0])
            print("image_shape = ", test_images.shape)
            print("image_type = ", test_images.dtype)
            #print("label[0] = \n", test_labels[0])
            print("label_shape = ",test_labels.shape)
            print("label_type = ", test_labels.dtype)
            #cv2.imshow("photo", train_labels[0])
            #cv2.waitKey (0)
        '''
        #test_images = self.eval_data

        '''
        print("----Start test----")
        for count, image in enumerate(test_images[:10]):
            predict_results = self.predictor({'x':np.expand_dims(image, 0)})["classes"][0]
            print(predict_results)
            
            fig = plt.figure(figsize=(10,10))
            plt.subplot(1,1,1)
            plt.imshow(np.reshape(image, (28,28))) # 顯示圖片
            #plt.subplot(1,2,2)
            #plt.imshow(cropped_image_with_box.astype(np.int)) # 顯示圖片
            plt.show()
        '''
            
        #    cv2.imshow('.', np.reshape(image, (28,28,1)))
        #    k = cv2.waitKey (0)
        #    if k == ord('Q'):
            #break
        #cv2.destroyAllWindows()

        '''
        for count, image in enumerate(test_images):
            predict_results = self.predictor({'x':np.expand_dims(image, 0)})["classes"][0]

            print("predicted_classes = ",self.CLASS[predict_results])

            img = np.zeros((image.shape[0]+200, image.shape[1], 3), np.uint8)

            # put text
            img_text = self.draw_text(img, self.CLASS[predict_results], (0, image.shape[0])).astype(np.float32)
            img_text = np.reshape(img_text*255, (img_text.shape[0], img_text.shape[1], 3))

            img_text[:image.shape[0]] = image[:]

            cv2.imshow('Output frame', img_text)
            #cv2.imshow('Output frame', img)
            k=cv2.waitKey(0)
            if k == ord('Q'):
                break
            #break

        cv2.destroyAllWindows()
        '''

        print("----End test----")