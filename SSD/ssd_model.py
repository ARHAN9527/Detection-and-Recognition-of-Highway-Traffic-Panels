from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf

from ssd_layers import *
from ssd_anchors import ssd_anchors_all_layers

# SSD parameters
SSDParams = namedtuple('SSDParameters', ['img_shape',  # the input image size: 300x300
                                         'num_classes',  # number of classes: 20+1
                                         'no_annotation_label',
                                         'feat_layers', # list of names of layer for detection
                                         'feat_shapes', # list of feature map sizes of layer for detection
                                         'anchor_size_bounds', # the down and upper bounds of anchor sizes
                                         'anchor_sizes',   # list of anchor sizes of layer for detection
                                         'anchor_ratios',  # list of rations used in layer for detection
                                         'anchor_steps',   # list of cell size (pixel size) of layer for detection
                                         'anchor_offset',  # the center point offset
                                         'normalizations', # list of normalizations of layer for detection
                                         'prior_scaling'   #
                                         ])

class SSD_model:
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)


        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layer4", "layer7", "layer8", "layer9", "layer10", "layer11"],
                                    feat_shapes=[(38, 38),
                                                 (19, 19),
                                                 (10, 10),
                                                 (5, 5),
                                                 (3, 3),
                                                 (1, 1)],
                                    anchor_size_bounds=[0.10, 0.90],  # diff from the original paper
                                    anchor_sizes=[(30., 78.),
                                                  (78., 126.),
                                                  (126., 174.),
                                                  (174., 222.),
                                                  (222., 270.),
                                                  (270., 318.)],
                                    anchor_ratios=[[2, .5],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5],
                                                   [2, .5]],
                                    anchor_steps=[8, 16, 30, 60, 100, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model.serving_input_receiver_fn)

    def _backbone(self, x, is_training):
        exp = 6
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3], name="input_tensor")
        
        # original vgg layers      
        # block 1
        net = conv2d(net, 64, 3, scope="conv1_1")
        net = conv2d(net, 64, 3, scope="conv1_2") # 300x300
        self.end_points["layer1"] = net
        net = max_pool2d(net, 2, scope="pool1")
        # block 2
        net = conv2d(net, 128, 3, scope="conv2_1")
        net = conv2d(net, 128, 3, scope="conv2_2") # 150x150
        self.end_points["layer2"] = net
        net = max_pool2d(net, 2, scope="pool2")
        # block 3        
        net = conv2d(net, 256, 3, scope="conv3_1")
        net = conv2d(net, 256, 3, scope="conv3_2")
        net = conv2d(net, 256, 3, scope="conv3_3") # 75x75
        self.end_points["layer3"] = net
        net = max_pool2d(net, 2, scope="pool3")
        # block 4
        net = conv2d(net, 512, 3, scope="conv4_1")
        net = conv2d(net, 512, 3, scope="conv4_2")
        net = conv2d(net, 512, 3, scope="conv4_3") # 38x38
        self.end_points["layer4"] = net
        net = max_pool2d(net, 2, scope="pool4")
        # block 5
        net = conv2d(net, 512, 3, scope="conv5_1")
        net = conv2d(net, 512, 3, scope="conv5_2")
        net = conv2d(net, 512, 3, scope="conv5_3") # 19x19
        self.end_points["layer5"] = net
        net = max_pool2d(net, 3, stride=1, scope="pool5")
        
        return net
        
    def _extra_layer(self, net, is_training):        
        ''' Additional SSD layers'''
        
        # block 6: use dilate conv
        net = conv2d(net, 1024, 3, dilation_rate=6, scope="conv6") # 19x19
        self.end_points["layer6"] = net
        #net = dropout(net, is_training=self.is_training)
        # block 7
        net = conv2d(net, 1024, 1, scope="conv7") # 19x19
        self.end_points["layer7"] = net
        # block 8
        net = conv2d(net, 256, 1, scope="conv8_1x1")
        net = conv2d(pad2d(net, 1), 512, 3, stride=2, scope="conv8_3x3", padding="valid") # 10x10
        self.end_points["layer8"] = net
        # block 9
        net = conv2d(net, 128, 1, scope="conv9_1x1")
        net = conv2d(pad2d(net, 1), 256, 3, stride=2, scope="conv9_3x3", padding="valid") # 5x5
        self.end_points["layer9"] = net
        # block 10
        net = conv2d(net, 128, 1, scope="conv10_1x1")
        net = conv2d(net, 256, 3, scope="conv10_3x3", padding="valid") # 3x3
        self.end_points["layer10"] = net
        # block 11
        net = conv2d(net, 128, 1, scope="conv11_1x1")
        net = conv2d(net, 256, 3, scope="conv11_3x3", padding="valid") # 1x1
        self.end_points["layer11"] = net
    
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

        net = self._backbone(features["x"], is_training)
        self._extra_layer(net, is_training)

        # class and location predictions
        predictions = []
        logits = []
        locations = []
        for i, layer in enumerate(self.ssd_params.feat_layers):
            cls_pred, loc_pred = ssd_multibox_layer(self.end_points[layer],
                                                    self.ssd_params.num_classes,
                                                    self.ssd_params.anchor_sizes[i],
                                                    self.ssd_params.anchor_ratios[i],
                                                    self.ssd_params.normalizations[i],
                                                    is_training=is_training,
                                                    scope=layer+"_box")
            logits.append(cls_pred)
            locations.append(loc_pred)
   
        return logits, locations

    def _bboxes_encode_layer(self,
                             labels,
                             bboxes,
                             anchors_layer,
                             num_classes,
                             no_annotation_label,
                             ignore_threshold=0.5,
                             prior_scaling=[0.1, 0.1, 0.2, 0.2],
                             dtype=tf.float32):
        """
        Encode groundtruth labels and bounding boxes using SSD anchors from
        one layer.

        Arguments:
        labels: 1D Tensor(int64) containing groundtruth labels;
        bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
        anchors_layer: Numpy array with layer anchors;
        matching_threshold: Threshold for positive match with groundtruth bboxes;
        prior_scaling: Scaling of encoded coordinates.

        Return:
        feat_labels:4D Tensor containing groundtruth labels [batch, size, size, num_anchor]
        feat_scores:4D Tensor containing groundtruth scores [batch, size, size, num_anchor]
        feat_localizations:4D Tensor containing groundtruth localizations [batch, size*size, num_anchor, 4]
        """
        batch_num = labels.get_shape()[0] # Batch
        # Anchors coordinates and volume.
        yref, xref, href, wref = anchors_layer
        ymin = yref - href / 2.
        xmin = xref - wref / 2.
        ymax = yref + href / 2.
        xmax = xref + wref / 2.

        # Batch
        reshape = lambda data_list,toshape: map(lambda data: tf.reshape(data,toshape), data_list)
        
        shape = list(ymin.shape)
        shape.insert(0,batch_num)
        batch_zero = tf.zeros((shape), dtype=dtype)
        ymin = tf.maximum(batch_zero, ymin) # batch,size,size,num_anchor
        xmin = tf.maximum(batch_zero, xmin)
        ymax = tf.maximum(batch_zero, ymax)
        xmax = tf.maximum(batch_zero, xmax)
        #####################################################

        # Initialize tensors...        
        #shape = (batch_num, yref.shape[0], yref.shape[1], href.size) # Single
        feat_labels = tf.zeros(shape, dtype=tf.int64) # batch,size,size,num_anchor
        feat_scores = tf.zeros(shape, dtype=dtype)

        feat_ymin = tf.zeros(shape, dtype=dtype)
        feat_xmin = tf.zeros(shape, dtype=dtype)
        feat_ymax = tf.ones(shape, dtype=dtype)
        feat_xmax = tf.ones(shape, dtype=dtype)        
        
        # Batch
        ymin, xmin, ymax, xmax, feat_labels, feat_scores,\
        feat_ymin, feat_xmin, feat_ymax, feat_xmax = reshape([ymin, 
                                                              xmin, 
                                                              ymax, 
                                                              xmax,
                                                              feat_labels, 
                                                              feat_scores, 
                                                              feat_ymin, 
                                                              feat_xmin, 
                                                              feat_ymax, 
                                                              feat_xmax], [batch_num,-1])        
        ########################################################################################################
        vol_anchors = (xmax - xmin) * (ymax - ymin) # batch,-1

        def jaccard_with_anchors(bbox):
            """Compute jaccard score between a box and the anchors."""           
            int_ymin = tf.maximum(ymin, bbox[0]) # batch,-1                                  
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.) # batch,-1
            w = tf.maximum(int_xmax - int_xmin, 0.)
            # Volumes.
            inter_vol = h * w
            union_vol = vol_anchors - inter_vol + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])            
            jaccard = tf.div(inter_vol, union_vol) # batch,-1
            return jaccard            

        def intersection_with_anchors(bbox):
            """Compute intersection between score a box and the anchors."""
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_vol = h * w
            scores = tf.div(inter_vol, vol_anchors)
            return scores

        def condition(i, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """Condition: check label index."""
            # Single
            #r = tf.less(i, tf.shape(labels))            
            # Batch            
            r = tf.less(i, tf.shape(labels[0]))            
            
            return r[0]            

        def body(i, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """
            Body: update feature labels, scores and bboxes.
            Follow the original SSD paper for that purpose:
            - assign values when jaccard > 0.5;
            - only update if beat the score of other bboxes.
            """
            # Jaccard score.
            # Single
            #label = labels[i] # 1
            #bbox = bboxes[i] # 4,
            # Batch
            label = labels[:,i] #batch,
            bbox = bboxes[:,i] #batch, 4            
            bbox = [i for i in map(lambda x: tf.reshape(x,[-1,1]), [bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]])]

            jaccard = jaccard_with_anchors(bbox)                           # batch,size,size,num_anchor
            # Mask: check threshold + scores + no annotations + num_classes.
            mask = tf.greater(jaccard, feat_scores)                        # batch,size,size,num_anchor
            matching_threshold=0.5
            mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
            #mask = tf.logical_and(mask, feat_scores > -0.5)
            #mask = tf.logical_and(mask, label < num_classes)              # Single
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, dtype)
            
            # Update values using mask.
            #feat_labels = imask * label + (1 - imask) * feat_labels       # Single      
            feat_labels = imask * tf.reshape(label,[-1,1]) + (1 - imask) * feat_labels      # Batch
            feat_scores = fmask * jaccard + (1 - fmask) * feat_scores
            #feat_scores = tf.where(mask, jaccard, feat_scores)
            feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
            feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
            feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
            feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

            # Check no annotation label: ignore these anchors...
            # interscts = intersection_with_anchors(bbox)
            # mask = tf.logical_and(interscts > ignore_threshold,
            #                       label == no_annotation_label)
            # # Replace scores by -1.
            # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

            return [i+1, feat_labels, feat_scores,
                    feat_ymin, feat_xmin, feat_ymax, feat_xmax]
        # Main loop definition.
        i = 0
        [i, feat_labels, feat_scores,
         feat_ymin, feat_xmin, feat_ymax, feat_xmax] = tf.while_loop(condition,
                                                                     body,
                                                                     [i, feat_labels, feat_scores,
                                                                      feat_ymin, feat_xmin, feat_ymax, feat_xmax])
        
        # Batch
        feat_labels, feat_scores, feat_ymin,\
        feat_xmin, feat_ymax, feat_xmax = reshape([feat_labels, 
                                                   feat_scores, 
                                                   feat_ymin, 
                                                   feat_xmin, 
                                                   feat_ymax, 
                                                   feat_xmax], shape) # [batch, size, size, num_anchor]
        
        # Transform to center / size.
        feat_cy = (feat_ymax + feat_ymin) / 2.
        feat_cx = (feat_xmax + feat_xmin) / 2.
        feat_h = feat_ymax - feat_ymin
        feat_w = feat_xmax - feat_xmin
        # Encode features.
        prior_scaling = self.ssd_params.prior_scaling

        feat_cy = (feat_cy - yref) / href / prior_scaling[0]
        feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
        feat_h = tf.log(feat_h / href) / prior_scaling[2]
        feat_w = tf.log(feat_w / wref) / prior_scaling[3]
        # Use SSD ordering: x / y / w / h instead of ours.        
        feat_cx, feat_cy, feat_w, feat_h = reshape([feat_cx, feat_cy, feat_w, feat_h], [shape[0], -1, shape[-1]])        
        
        feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1) # [batch, size*size, num_anchor, 4]
        
        return feat_labels, feat_localizations, feat_scores # [batch, size, size, num_anchor]

    def _bboxes_encode(self,
                       labels,
                       bboxes,
                       anchors,
                       num_classes,
                       no_annotation_label,
                       ignore_threshold=0.5,
                       prior_scaling=[0.1, 0.1, 0.2, 0.2],
                       dtype=tf.float32,
                       scope='ssd_bboxes_encode'):
        """
        Encode groundtruth labels and bounding boxes using SSD net anchors.
        Encoding boxes for all feature layers.

        Arguments:
        labels: 1D Tensor(int64) containing groundtruth labels;
        bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
        anchors: List of Numpy array with layer anchors;
        matching_threshold: Threshold for positive match with groundtruth bboxes;
        prior_scaling: Scaling of encoded coordinates.

        Return:
        (target_labels, target_localizations, target_scores):
            Each element is a list of target Tensors.
        """
        with tf.name_scope(scope):
            target_labels = []
            target_localizations = []
            target_scores = []
            for i, anchors_layer in enumerate(anchors):
                with tf.name_scope('bboxes_encode_block_%i' % i):
                    t_labels, t_loc, t_scores = self._bboxes_encode_layer(labels,
                                                                          bboxes,
                                                                          anchors_layer,
                                                                          num_classes,
                                                                          ignore_threshold,
                                                                          self.ssd_params.prior_scaling,
                                                                          dtype)
                    target_labels.append(t_labels)
                    target_localizations.append(t_loc)
                    target_scores.append(t_scores)
            return target_labels, target_localizations, target_scores

    def bboxes_encode(self,
                      labels,
                      bboxes,
                      anchors,
                      scope=None):
        """
        Encode labels and bounding boxes.
        """
        return self._bboxes_encode(labels,
                                   bboxes,
                                   anchors,
                                   self.ssd_params.num_classes,
                                   self.ssd_params.no_annotation_label,
                                   ignore_threshold=0.5,
                                   prior_scaling=self.ssd_params.prior_scaling,
                                   scope=scope)

    def _bboxes_decode_layer(self,
                             feat_localizations,
                             anchors_layer,
                             prior_scaling=[0.1, 0.1, 0.2, 0.2]):
        """
        Compute the relative bounding boxes from the layer features and
        reference anchor bounding boxes.

        Arguments:
        feat_localizations: Tensor containing localization features.
                            5D Tensor, [batch_size, size, size, n_anchors, 4]
        anchors: List of numpy array containing anchor boxes.
                 list of Tensors(y, x, w, h)
                 shape: [size,size,1], [size, size,1], [n_anchors], [n_anchors]
        prior_scaling: list of 4 floats

        Return:
        Tensor Nx4: ymin, xmin, ymax, xmax
                    [batch_size, size, size, n_anchors, 4]
        """
        yref, xref, href, wref = anchors_layer
        
        xref = tf.reshape(xref, [-1, 1])
        yref = tf.reshape(yref, [-1, 1])        

        # Compute center, height and width
        cx = feat_localizations[:, :, :, 0] * wref * prior_scaling[0] + xref
        cy = feat_localizations[:, :, :, 1] * href * prior_scaling[1] + yref
        w = wref * tf.exp(feat_localizations[:, :, :, 2] * prior_scaling[2])
        h = href * tf.exp(feat_localizations[:, :, :, 3] * prior_scaling[3])
        # Boxes coordinates.
        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1) # [batch, size*size, num_anchor, 4]
        
        return bboxes

    def _bboxes_decode(self,
                       feat_localizations,
                       anchors,
                       prior_scaling=[0.1, 0.1, 0.2, 0.2],
                       scope='ssd_bboxes_decode'):
        """Compute the relative bounding boxes from the SSD net features and
        reference anchors bounding boxes.

        Arguments:
        feat_localizations: List of Tensors containing localization features.
        anchors: List of numpy array containing anchor boxes.

        Return:
        List of Tensors Nx4: ymin, xmin, ymax, xmax
        """
        with tf.name_scope(scope):
            bboxes = []
            for i, anchors_layer in enumerate(anchors):
                anchors_tlayer = list(map(tf.convert_to_tensor, anchors_layer))
                bboxes.append(self._bboxes_decode_layer(feat_localizations[i],
                                                        anchors_tlayer,
                                                        prior_scaling))
            return bboxes

    def bboxes_decode(self,
                      feat_localizations,
                      anchors,
                      scope='ssd_bboxes_decode'):
        """
        Compute the relative bounding boxes from the SSD net features and
        reference anchors bounding boxes.
        """
        return self._bboxes_decode(feat_localizations,
                                   anchors,
                                   prior_scaling=self.ssd_params.prior_scaling,
                                   scope=scope)    
    
    def _bboxes_select_layer(self, feat_predictions, feat_locations):
        """Select boxes from the feat layer, only for bacth_size=1"""
        #n_bboxes = np.product(feat_predictions.get_shape().as_list()[:-1])
        # decode the location
        # locations [batch_size, size, size, n_anchors, 4]
        locations = feat_locations
        #locations = self._bboxes_decode_layer(feat_locations, anchor_bboxes, prior_scaling)
        locations = tf.reshape(locations, [-1, 4])
        softmax_predictions = tf.nn.softmax(feat_predictions)
        predictions = tf.reshape(softmax_predictions, [-1, self.ssd_params.num_classes])
        # remove the background predictions
        sub_predictions = predictions[:, 1:]
        # choose the max score class
        classes = tf.argmax(sub_predictions, axis=1) + 1  # class labels
        scores = tf.reduce_max(sub_predictions, axis=1)   # max_class scores
        # Boxes selection: use threshold
        '''
        filter_mask = scores > self.threshold
        classes = tf.boolean_mask(classes, filter_mask) ################################ TODO
        scores = tf.boolean_mask(scores, filter_mask)
        locations = tf.boolean_mask(locations, filter_mask)
        '''
        '''
        filter_mask = scores > self.threshold
        indices = tf.where(filter_mask) # 2D [num, coordinates]
        indices = tf.squeeze(indices) # 1D 
        classes = tf.gather(classes, indices)
        scores = tf.gather(scores, indices)
        locations = tf.gather(locations, indices)
        '''
        return classes, scores, locations
    
        '''
        n_bboxes = np.product(feat_predictions.get_shape().as_list()[1:-1])
        # decode the location
        # locations [batch_size, size, size, n_anchors, 4]
        locations = feat_locations
        #locations = self._bboxes_decode_layer(feat_locations, anchor_bboxes, prior_scaling)
        locations = tf.reshape(locations, [n_bboxes, 4])
        softmax_predictions = tf.nn.softmax(feat_predictions)
        predictions = tf.reshape(softmax_predictions, [n_bboxes, self.ssd_params.num_classes])
        # remove the background predictions
        sub_predictions = predictions[:, 1:]
        # choose the max score class
        classes = tf.argmax(sub_predictions, axis=1) + 1  # class labels
        scores = tf.reduce_max(sub_predictions, axis=1)   # max_class scores
        # Boxes selection: use threshold
        filter_mask = scores > self.threshold
        classes = tf.boolean_mask(classes, filter_mask)
        scores = tf.boolean_mask(scores, filter_mask)
        locations = tf.boolean_mask(locations, filter_mask)
        return classes, scores, locations
        '''
        
    def _bboxes_select(self, predictions, locations):
        """Select all bboxes predictions, only for bacth_size=1"""
        #anchor_bboxes_list = self.anchor_bboxes_list
        classes_list = []
        scores_list = []
        bboxes_list = []
        # select bboxes for each feat layer
        for n in range(len(predictions)):
            #anchor_bboxes = list(map(tf.convert_to_tensor, anchor_bboxes_list[n]))
            classes, scores, bboxes = self._bboxes_select_layer(predictions[n],
                                                                locations[n])
            classes_list.append(classes)
            scores_list.append(scores)
            bboxes_list.append(bboxes)
        # combine all feat layers
        classes = tf.concat(classes_list, axis=0, name="output_class")
        scores = tf.concat(scores_list, axis=0, name="output_scores")
        bboxes = tf.concat(bboxes_list, axis=0, name="output_bboxes")

        # classes, scores [n_bboxes]
        # bboxes [n_bboxes, 4]
        return classes, scores, bboxes

    def _abs_smooth(self, x):
        """Smoothed absolute function. Useful to compute an L1 smooth error.

        Define as:
            x^2 / 2         if abs(x) < 1
            abs(x) - 0.5    if abs(x) > 1
        We use here a differentiable definition using min(x) and abs(x). Clearly
        not optimal, but good enough for our purpose!
        """
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r

    def _focal_loss_softmax(self, logits, labels, alpha=0.00000025, gamma=10):
        """
        Computer focal loss for multi classification
        Args:
          labels: A int32 tensor of shape [batch_size].
          logits: A float32 tensor of shape [batch_size,num_classes].
          gamma: A scalar for focal loss gamma hyper-parameter.
        Returns:
          A tensor of the same shape as `lables`
        """
        y_pred = tf.nn.softmax(logits) # [batch_size,num_classes]
        labels = tf.one_hot(labels, depth=y_pred.shape[1])
        loss = -labels * (alpha*(1-y_pred)**gamma) * tf.log(y_pred)
        #loss = tf.reduce_max(loss, axis=1)
        loss = tf.reduce_sum(loss)
        return loss

    def _losses(self,
                logits,
                localisations,
                gclasses,
                glocalisations,
                gscores,
                match_threshold=0.5,
                negative_ratio=3.,
                alpha=1.,
                label_smoothing=0.,
                device='/cpu:0',
                scope=None):
        with tf.name_scope(scope, 'ssd_losses'):
            lshape = logits[0].get_shape().as_list()
            #lshape = tfe.get_shape(logits[0], 5)            
            num_classes = lshape[-1]
            batch_size = lshape[0]

            # Flatten out all vectors!
            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []
            for i in range(len(logits)):
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))

            # And concat the crap!
            logits = tf.concat(flogits, axis=0)
            gclasses = tf.concat(fgclasses, axis=0)
            gscores = tf.concat(fgscores, axis=0)
            localisations = tf.concat(flocalisations, axis=0)
            glocalisations = tf.concat(fglocalisations, axis=0)
            dtype = logits.dtype

            #loss_p = self._focal_loss_softmax(logits=logits, labels=gclasses)            
            
            # Compute positive matching mask...
            pmask = gscores > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)
            
            # Hard negative mining...
            no_classes = tf.cast(pmask, tf.int32)
            predictions = tf.nn.softmax(logits, name="loss_tensor")
            #predictions = slim.softmax(logits)            
            nmask = tf.logical_not(pmask)
            #nmask = tf.logical_and(nmask, gscores > -0.5)
            #nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)
            fnmask = tf.cast(nmask, dtype)
            nvalues = fnmask * predictions[:, 0] + (1 - fnmask) * fnmask # replace where
            #nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
            nvalues_flat = tf.reshape(nvalues, [-1])
            # Number of negative entries to select.
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positives, tf.int32) #+ batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)
            n_neg = tf.maximum(n_neg, 1)

            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            max_hard_pred = -val[-1]
            # Final negative mask.
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
            fnmask = tf.cast(nmask, dtype)

            n = n_positives + tf.cast(n_neg, tf.float32)
            
            # Add cross-entropy loss.
            loss_p = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)            
            loss_p = tf.div(tf.reduce_sum(loss_p * fpmask), n, name='value_p')

            loss_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
            loss_n = tf.div(tf.reduce_sum(loss_n * fnmask), n, name='value_n')

            # Add localization loss: smooth L1, L2, ...
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss_loc = self._abs_smooth(localisations - glocalisations)
            loss_loc = tf.div(tf.reduce_sum(loss_loc * weights), n, name='value_loc')

            return loss_p + loss_n + loss_loc
            #return loss_p + loss_loc
            
    def losses(self,
               logits,
               localisations,
               gclasses,
               glocalisations,
               gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """
        Define the SSD network losses.
        """
        return self._losses(logits,
                            localisations,
                            gclasses,
                            glocalisations,
                            gscores,
                            match_threshold=match_threshold,
                            negative_ratio=negative_ratio,
                            alpha=alpha,
                            label_smoothing=label_smoothing,
                            scope=scope)

    def model_fn(self, features, labels, mode):
        # logits [n_feat_layer, batch, size, size, n_anchors, n_classes]
        # locations [n_feat_layer, batch, size, size, n_anchors, 4]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        logits, locations = self._built_net(features, is_training)        
        # =========================================================================== #
        # PREDICT mode
        # =========================================================================== #
        if mode == tf.estimator.ModeKeys.PREDICT:
            locations_decode = self.bboxes_decode(locations, self.anchor_bboxes_list)
            # classes, scores, bboxes: [n_boxes], [n_boxes], [n_boxes, 4]
            classes, scores, bboxes = self._bboxes_select(logits, locations_decode)
            
            #classes = tf.cast(classes, tf.float32)
            #classes = tf.expand_dims(classes, -1)
            scores = tf.expand_dims(scores, -1)
            tfoutput = tf.concat((scores, bboxes), axis=-1, name="tfoutput") #########################################
 
            '''
            selected_indices = tf.image.non_max_suppression(boxes=bboxes,
                                                            scores=scores,
                                                            max_output_size=10,
                                                            iou_threshold=0.5,
                                                            name = "nms")

            classes = tf.gather(classes,selected_indices, name="nms_class")
            scores = tf.gather(scores,selected_indices, name="nms_score")
            bboxes = tf.gather(bboxes,selected_indices)

            bboxes = tf.transpose(bboxes)
            b0 = tf.expand_dims(tf.maximum(bboxes[0], 0), axis=0)
            b1 = tf.expand_dims(tf.maximum(bboxes[1], 0), axis=0)
            b2 = tf.expand_dims(tf.minimum(bboxes[2], 1), axis=0)
            b3 = tf.expand_dims(tf.minimum(bboxes[3], 1), axis=0)
            bboxes = tf.transpose(tf.concat((b0, b1, b2, b3), axis=0), name="nms_bbox")
            '''
            
            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "tfoutput": tfoutput,
            }
            '''
            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": classes,
                "scores": scores,
                "bboxes": bboxes,
                "nbox": nbox,
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                #"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            '''

            export_outputs = {"predictions": tf.estimator.export.PredictOutput(predictions)}
            #export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs['pred']
            export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs['predictions']
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs)

        # Calculate Loss (for both TRAIN and EVAL modes)
        # Single
        #cla = labels['class'][0]
        #bbox = labels['bbox'][0]
        predict_probabilities = {"probabilities": tf.nn.softmax(logits[5][0], name="softmax_tensor")}
        # Batch
        cla = labels['class']
        bbox = labels['bbox']
        # gclasses, gscores: anchor_layer [size, size, n_anchor, ]
        # glocalisations: anchor_layer [size, size, n_anchor, 4]
        gclasses, glocalisations, gscores = self.bboxes_encode(cla, bbox, self.anchor_bboxes_list)

        loss = self.losses(logits,
                           locations,
                           gclasses,
                           glocalisations,
                           gscores,
                           match_threshold=0.5,
                           negative_ratio=3,
                           alpha=1,
                           label_smoothing=0.0)

        # =========================================================================== #
        # TRAIN mode
        # =========================================================================== #
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(1e-3, tf.train.get_global_step(), 60000, 0.1, staircase = True)
            optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss=loss,
                                          global_step=tf.train.get_global_step())
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#############################################
            train_op = tf.group([train_op, update_ops])#########################################################

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)

        # =========================================================================== #
        # EVAL mode
        # =========================================================================== #
        # Add evaluation metrics (for EVAL mode)
        '''
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                           predictions=predictions["classes"])}

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
        '''

    def train(self, dataset_path = "./frame/panels/npz/train", data = None, step=50):
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
        train_labels = {'class': np.array(data["class"], np.int64), 'bbox': np.array(data["bbox"], np.float32)}

        #print("label_shape = ",train_labels.shape)
        #print("label_type = ", train_labels.dtype)

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

        self.ssd_estimator.train(input_fn=train_input_fn,
                                 steps=step,
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

        eval_results = self.ssd_estimator.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        print("----End eval----")

    def serving_input_receiver_fn():
        x = tf.placeholder(dtype=tf.float32, shape=[1, 300, 300, 3], name='x')
        features = receiver_tensors = {'x': x}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    def save_model(self, dir_path='saved_model'):        
        self.ssd_estimator.export_savedmodel(dir_path, SSD_model.serving_input_receiver_fn)
    
    def test(self, data=None, dataset_path = "./frame/panels/npz/train"):
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

        #test_images = data["img"]

        '''
        print("----Start test----") 
        pred = self.predictor({'x':data["img"]})
        print("finish predict ")
        '''
        return self.predictor(data)
        
        '''
        pred = []
        for i, d in enumerate(data):
            test_images = d["img"]
            #pred = self.predictor({'x':test_images})
            pred.append(self.predictor({'x':test_images}))
            print("finish predict ", i)
        '''

        #for count, image in enumerate(test_images):
        #    predict_results = self.predictor({'x':np.expand_dims(image, 0)})#["classes"][0]
        #    print(predict_results)
        #    cv2.imshow('.', np.reshape(image, (28,28,1)))
        #    k = cv2.waitKey (0)
        #    if k == ord('Q'):
        #    break
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
        return pred

    
WIDTH_MULTIPLIER = 0.125
#########################################################################################################################################
class SSD_model_vgg(SSD_model):
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)


        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layer6", "layer7", "layer12", "layer13", "layer14", "layer15"],
                                    feat_shapes=[(19, 19),
                                                 (10, 10),
                                                 (5, 5),
                                                 (3, 3),
                                                 (2, 2),
                                                 (1, 1)],
                                    anchor_size_bounds=[0.10, 0.90],  # diff from the original paper
                                    anchor_sizes=[(30., 78.),
                                                  (78., 126.),
                                                  (126., 174.),
                                                  (174., 222.),
                                                  (222., 270.),
                                                  (270., 318.)],
                                    anchor_ratios=[[.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75]],
                                    anchor_steps=[16, 30, 60, 100, 150, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model_mobilev2.serving_input_receiver_fn)

    def _backbone(self, x, is_training):
        width_multiplier = 1
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3], name="input_tensor")
        
        # original vgg layers      
        # block 1
        net = conv2d(net, 64, 3, scope="conv1_1")
        net = conv2d(net, 64, 3, scope="conv1_2") # 300x300
        self.end_points["layer1"] = net
        net = max_pool2d(net, 2, scope="pool1")
        # block 2
        net = conv2d(net, 128, 3, scope="conv2_1")
        net = conv2d(net, 128, 3, scope="conv2_2") # 150x150
        self.end_points["layer2"] = net
        net = max_pool2d(net, 2, scope="pool2")
        # block 3        
        net = conv2d(net, 256, 3, scope="conv3_1")
        net = conv2d(net, 256, 3, scope="conv3_2")
        net = conv2d(net, 256, 3, scope="conv3_3") # 75x75
        self.end_points["layer3"] = net
        net = max_pool2d(net, 2, scope="pool3")
        # block 4
        net = conv2d(net, 512, 3, scope="conv4_1")
        net = conv2d(net, 512, 3, scope="conv4_2")
        net = conv2d(net, 512, 3, scope="conv4_3") # 38x38
        self.end_points["layer4"] = net
        net = max_pool2d(net, 2, scope="pool4")
        # block 5
        net = conv2d(net, 512, 3, scope="conv5_1")
        net = conv2d(net, 512, 3, scope="conv5_2")
        net = conv2d(net, 512, 3, scope="conv5_3") # 19x19
        self.end_points["layer5"] = net
        net = max_pool2d(net, 3, stride=1, scope="pool5")
        # block 6: use dilate conv
        net = conv2d(net, 1024, 3, dilation_rate=6, scope="conv6") # 19x19
        self.end_points["layer6"] = net
        #net = dropout(net, is_training=self.is_training)
        # block 7
        net = conv2d(pad2d(net, 1), 1024, 3, 2, scope="conv7", padding="valid") # 10x10
        self.end_points["layer7"] = net        
        
        return net
        
    def _extra_layer(self, net, is_training):
        ''' Additional SSD layers'''
        
        # layer12: input[batch,10,10,1024], output[batch,5,5,512]
        net = ssd_conv2d(net, 256, 2, is_training, scope="ssd_conv2d12_1", padding="same") # 5
        self.end_points["layer12"] = net
        # layer13: input[batch,5,5,512], output[batch,3,3,256]
        net = ssd_conv2d(net, 128, 1, is_training, scope="ssd_conv2d13_1") # 3
        self.end_points["layer13"] = net
        # layer14: input[batch,3,3,256], output[batch,2,2,256]
        net = ssd_conv2d(pad2d(net, 1), 128, 2, is_training, scope="ssd_conv2d14_1") # 2
        self.end_points["layer14"] = net
        # layer14: input[batch,2,2,256], output[batch,1,1,256]
        net = ssd_conv2d(pad2d_single(net, 1), 128, 1, is_training, scope="ssd_conv2d15_1") # 1
        self.end_points["layer15"] = net
        
        print(self.end_points) 
    
#########################################################################################################################################
class SSD_model_mobilev1(SSD_model):
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)

        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layer5", "layer6", "layer12", "layer13", "layer14", "layer15"],
                                    feat_shapes=[(19, 19),
                                                 (10, 10),
                                                 (5, 5),
                                                 (3, 3),
                                                 (2, 2),
                                                 (1, 1)],
                                    anchor_size_bounds=[0.10, 0.90],  # diff from the original paper
                                    anchor_sizes=[(30., 78.),
                                                  (78., 126.),
                                                  (126., 174.),
                                                  (174., 222.),
                                                  (222., 270.),
                                                  (270., 318.)],
                                    anchor_ratios=[[.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75]],
                                    anchor_steps=[16, 30, 60, 100, 150, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model_mobilev2.serving_input_receiver_fn)

    def _backbone(self, x, is_training):
        width_multiplier = WIDTH_MULTIPLIER
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3], name="input_tensor")     
        
        # mobilev1
        # layer1: input[batch,300,300,3], output[batch,150,150,32]
        net = conv2d(net, 32, 3, 2, scope="conv1_1") # size/2 150
        self.end_points["layer1"] = net
        # layer2: input[batch,150,150,32], output[batch,150,150,64]
        net = depth_point_conv2d(net, 64*width_multiplier, 1, is_training, scope="dpconv2_1")
        self.end_points["layer2"] = net
        # layer3: input[batch,150,150,64], output[batch,75,75,128]
        net = depth_point_conv2d(net, 128*width_multiplier, 2, is_training, scope="dpconv3_1") # size/2 75
        net = depth_point_conv2d(net, 128*width_multiplier, 1, is_training, scope="dpconv3_2")
        self.end_points["layer3"] = net
        # layer4: input[batch,75,75,128], output[batch,38,38,256]
        net = depth_point_conv2d(pad2d(net, 1), 256*width_multiplier, 2, is_training, scope="dpconv4_1", padding="valid") # size/2 38
        net = depth_point_conv2d(net, 256*width_multiplier, 1, is_training, scope="dpconv4_2")
        self.end_points["layer4"] = net
        # layer5: input[batch,38,38,256], output[batch,19,19,512]
        net = depth_point_conv2d(net, 512*width_multiplier, 2, is_training, scope="dpconv5_1") # size/2 19
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_2")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_3")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_4")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_5")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_6")
        self.end_points["layer5"] = net
        # layer6: input[batch,19,19,512], output[batch,10,10,1024]
        net = depth_point_conv2d(pad2d(net, 1), 1024*width_multiplier, 2, is_training, scope="dpconv6_1", padding="valid") # size/2 10
        self.end_points["layer6"] = net
        # layer7: input[batch,10,10,1024], output[batch,10,10,1024]
        #net = depth_point_conv2d(net, 1024, 1, is_training, scope="dpconv7_1")
        #self.end_points["layer7"] = net
        
        return net
        
    def _extra_layer(self, net, is_training):
        ''' Additional SSD layers'''
        # layer12: input[batch,10,10,1024], output[batch,5,5,512]
        net = ssd_conv2d(net, 256, 2, is_training, scope="ssd_conv2d12_1", padding="same") # 5
        self.end_points["layer12"] = net
        # layer13: input[batch,5,5,512], output[batch,3,3,256]
        net = ssd_conv2d(net, 128, 1, is_training, scope="ssd_conv2d13_1") # 3
        self.end_points["layer13"] = net
        # layer14: input[batch,3,3,256], output[batch,2,2,256]
        net = ssd_conv2d(pad2d(net, 1), 128, 2, is_training, scope="ssd_conv2d14_1") # 2
        self.end_points["layer14"] = net
        # layer14: input[batch,2,2,256], output[batch,1,1,256]
        net = ssd_conv2d(pad2d_single(net, 1), 128, 1, is_training, scope="ssd_conv2d15_1") # 1
        self.end_points["layer15"] = net
        
        print(self.end_points)
######################################################################################################################################

class SSD_model_mobilev1_38(SSD_model):
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)


        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layer4", "layer6", "layer12", "layer13", "layer14", "layer15"],
                                    feat_shapes=[(38, 38),
                                                 (19, 19),
                                                 (10, 10),
                                                 (5, 5),
                                                 (3, 3),                                                 
                                                 (1, 1)],
                                    anchor_size_bounds=[0.10, 0.90],  # diff from the original paper
                                    anchor_sizes=[(30., 78.),
                                                  (78., 126.),
                                                  (126., 174.),
                                                  (174., 222.),
                                                  (222., 270.),
                                                  (270., 318.)],
                                    anchor_ratios=[[.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75]],
                                    anchor_steps=[8, 16, 30, 60, 100, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model_mobilev2.serving_input_receiver_fn)

    def _backbone(self, x, is_training):
        width_multiplier = WIDTH_MULTIPLIER
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3], name="input_tensor")     
        
        # mobilev1
        # layer1: input[batch,300,300,3], output[batch,150,150,32]
        net = conv2d(net, 32, 3, 2, scope="conv1_1") # size/2 150
        self.end_points["layer1"] = net
        # layer2: input[batch,150,150,32], output[batch,150,150,64]
        net = depth_point_conv2d(net, 64*width_multiplier, 1, is_training, scope="dpconv2_1")
        self.end_points["layer2"] = net
        # layer3: input[batch,150,150,64], output[batch,75,75,128]
        net = depth_point_conv2d(net, 128*width_multiplier, 2, is_training, scope="dpconv3_1") # size/2 75
        net = depth_point_conv2d(net, 128*width_multiplier, 1, is_training, scope="dpconv3_2")
        self.end_points["layer3"] = net
        # layer4: input[batch,75,75,128], output[batch,38,38,256]
        net = depth_point_conv2d(pad2d(net, 1), 256*width_multiplier, 2, is_training, scope="dpconv4_1", padding="valid") # size/2 38
        net = depth_point_conv2d(net, 256*width_multiplier, 1, is_training, scope="dpconv4_2")
        self.end_points["layer4"] = net
        # layer5: input[batch,38,38,256], output[batch,19,19,512]
        net = depth_point_conv2d(net, 512*width_multiplier, 2, is_training, scope="dpconv5_1") # size/2 19
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_2")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_3")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_4")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_5")
        net = depth_point_conv2d(net, 512*width_multiplier, 1, is_training, scope="dpconv5_6")
        self.end_points["layer5"] = net
        # layer6: input[batch,19,19,512], output[batch,10,10,1024]
        net = depth_point_conv2d(net, 1024*width_multiplier, 1, is_training, scope="dpconv6_1") # size/2 19
        self.end_points["layer6"] = net
        # layer7: input[batch,10,10,1024], output[batch,10,10,1024]
        #net = depth_point_conv2d(net, 1024, 1, is_training, scope="dpconv7_1")
        #self.end_points["layer7"] = net
        
        return net
        
    def _extra_layer(self, net, is_training):
        ''' Additional SSD layers'''
        # layer12: input[batch,10,10,1024], output[batch,5,5,512]
        net = ssd_conv2d(pad2d(net, 1), 256, 2, is_training, scope="ssd_conv2d12_1") # 10
        self.end_points["layer12"] = net
        # layer13: input[batch,5,5,512], output[batch,3,3,256]
        net = ssd_conv2d(net, 128, 2, is_training, scope="ssd_conv2d13_1", padding="same") # 5
        self.end_points["layer13"] = net
        # layer14: input[batch,3,3,256], output[batch,2,2,256]
        net = ssd_conv2d(net, 128, 1, is_training, scope="ssd_conv2d14_1") # 3
        self.end_points["layer14"] = net
        # layer14: input[batch,2,2,256], output[batch,1,1,256]
        net = ssd_conv2d(net, 128, 1, is_training, scope="ssd_conv2d15_1") # 1
        self.end_points["layer15"] = net
        
        print(self.end_points)
        
#########################################################################################################################################
class SSD_model_mobilev2(SSD_model):
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)


        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layer6", "layer8", "layer12", "layer13", "layer14", "layer15"],
                                    feat_shapes=[(19, 19),
                                                 (10, 10),
                                                 (5, 5),
                                                 (3, 3),
                                                 (2, 2),
                                                 (1, 1)],                                    
                                    anchor_size_bounds=[0.10, 0.90],  # diff from the original paper
                                    anchor_sizes=[(30., 78.),
                                                  (78., 126.),
                                                  (126., 174.),
                                                  (174., 222.),
                                                  (222., 270.),
                                                  (270., 318.)],
                                    anchor_ratios=[[.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75]],
                                    anchor_steps=[16, 30, 60, 100, 150, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model_mobilev2.serving_input_receiver_fn)

    def _backbone(self, x, is_training):
        exp = 6
        width_multiplier = WIDTH_MULTIPLIER
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3], name="input_tensor")
        
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
        '''
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
        '''
        ####### 9. 1个 1*1点卷积块 = 1*1 PW点卷积 +  BN + RELU6 1280个通道输出 ################
        #net = conv_bn_relu(net, 1280, 1, 1, is_training, scope="layer9_1") 
        #self.end_points["layer9"] = net
        ####### 10. 全局均值 池化 average_pooling2d #########################################
        #net = avg_pool2d(net, scope="avg_10_1")
        #self.end_points["block10"] = net
        ####### 11. 1*1 PW点卷积 后 展开成1维 ###############################################     
        #net = conv2d(net, 10, 1, activation=None, scope="conv11_1")
        #net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
        #self.end_points["block11"] = net
        
        return net
        
    def _extra_layer(self, net, is_training):
        '''
        net = tf.concat([upsample_conv2d(pad2d(self.end_points["layer4"], 1), 256, upsampling=False, scope='upsample9_1'),
                         upsample_conv2d(self.end_points["layer6"], 256, (40,40), scope='upsample9_2'),
                         upsample_conv2d(self.end_points["layer8"], 256, (40,40), scope='upsample9_3')], -1)
        
        net = batch_norm(net, training=is_training, scope="batch_norm9_4")
        self.end_points["layer9"] = net
        '''
        
        # additional SSD layers        
        '''
        net = ssd_conv2d(net, 512, 1, is_training, scope="ssd_conv2d10_1") # 38*38
        self.end_points["layer10"] = net
        net = ssd_conv2d(pad2d(net, 1), 512, 2, is_training, scope="ssd_conv2d11_1") # 19*19
        self.end_points["layer11"] = net
        '''
        # layer12: input[batch,10,10,1024], output[batch,5,5,512]
        net = ssd_conv2d(net, 256, 2, is_training, scope="ssd_conv2d12_1", padding="same") # 5
        self.end_points["layer12"] = net
        # layer13: input[batch,5,5,512], output[batch,3,3,256]
        net = ssd_conv2d(net, 128, 1, is_training, scope="ssd_conv2d13_1") # 3
        self.end_points["layer13"] = net
        # layer14: input[batch,3,3,256], output[batch,2,2,256]
        net = ssd_conv2d(pad2d(net, 1), 128, 2, is_training, scope="ssd_conv2d14_1") # 2
        self.end_points["layer14"] = net
        # layer14: input[batch,2,2,256], output[batch,1,1,256]
        net = ssd_conv2d(pad2d_single(net, 1), 128, 1, is_training, scope="ssd_conv2d15_1") # 1
        self.end_points["layer15"] = net
        
        print(self.end_points)

#########################################################################################################################################
class SSD_model_mobilev2_38(SSD_model):
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)

        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layer6", "layer8", "layer12", "layer13", "layer14", "layer15"],
                                    feat_shapes=[(38, 38),
                                                 (19, 19),
                                                 (10, 10),
                                                 (5, 5),
                                                 (3, 3),
                                                 (1, 1)],
                                    anchor_size_bounds=[0.10, 0.90],  # diff from the original paper
                                    anchor_sizes=[(30., 78.),
                                                  (78., 126.),
                                                  (126., 174.),
                                                  (174., 222.),
                                                  (222., 270.),
                                                  (270., 318.)],
                                    anchor_ratios=[[.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75]],
                                    anchor_steps=[8, 16, 30, 60, 100, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model_mobilev2.serving_input_receiver_fn)

    def _backbone(self, x, is_training):
        exp = 6
        width_multiplier = WIDTH_MULTIPLIER
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3], name="input_tensor")
        
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
        net = res_block(net, exp, 64*width_multiplier, 1, is_training, scope='res5_1') # size/16 38
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
        net = res_block(net, exp, 160*width_multiplier, 2, is_training, scope='res7_1', padding="same") # size/32 19
        net = res_block(net, exp, 160*width_multiplier, 1, is_training, scope='res7_2')
        net = res_block(net, exp, 160*width_multiplier, 1, is_training, scope='res7_3')
        self.end_points["layer7"] = net
        # layer8: input[batch,10,10,160], output[batch,10,10,320]
        net = res_block(net, exp, 320*width_multiplier, 1, is_training, shortcut=False, scope='res8_1') # 不进行残差合并 f(x)
        self.end_points["layer8"] = net
        
        ####### 9. 1个 1*1点卷积块 = 1*1 PW点卷积 +  BN + RELU6 1280个通道输出 ################
        #net = conv_bn_relu(net, 1280, 1, 1, is_training, scope="layer9_1") 
        #self.end_points["layer9"] = net
        ####### 10. 全局均值 池化 average_pooling2d #########################################
        #net = avg_pool2d(net, scope="avg_10_1")
        #self.end_points["block10"] = net
        ####### 11. 1*1 PW点卷积 后 展开成1维 ###############################################     
        #net = conv2d(net, 10, 1, activation=None, scope="conv11_1")
        #net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
        #self.end_points["block11"] = net
        
        return net
        
    def _extra_layer(self, net, is_training):
        '''
        net = tf.concat([upsample_conv2d(pad2d(self.end_points["layer4"], 1), 256, upsampling=False, scope='upsample9_1'),
                         upsample_conv2d(self.end_points["layer6"], 256, (40,40), scope='upsample9_2'),
                         upsample_conv2d(self.end_points["layer8"], 256, (40,40), scope='upsample9_3')], -1)
        
        net = batch_norm(net, training=is_training, scope="batch_norm9_4")
        self.end_points["layer9"] = net
        '''
        
        # additional SSD layers        
        '''
        net = ssd_conv2d(net, 512, 1, is_training, scope="ssd_conv2d10_1") # 38*38
        self.end_points["layer10"] = net
        net = ssd_conv2d(pad2d(net, 1), 512, 2, is_training, scope="ssd_conv2d11_1") # 19*19
        self.end_points["layer11"] = net
        '''
        # layer12: input[batch,10,10,320], output[batch,5,5,512]
        net = ssd_conv2d(pad2d(net, 1), 256, 2, is_training, scope="ssd_conv2d12_1", padding="valid") # 10
        self.end_points["layer12"] = net
        # layer13: input[batch,5,5,512], output[batch,3,3,256]
        net = ssd_conv2d(net, 128, 2, is_training, scope="ssd_conv2d13_1", padding="same") # 5
        self.end_points["layer13"] = net
        # layer14: input[batch,3,3,256], output[batch,2,2,256]
        net = ssd_conv2d(net, 128, 1, is_training, scope="ssd_conv2d14_1", padding="valid") # 3
        self.end_points["layer14"] = net
        # layer14: input[batch,2,2,256], output[batch,1,1,256]
        net = ssd_conv2d(net, 128, 1, is_training, scope="ssd_conv2d15_1") # 1
        self.end_points["layer15"] = net
        
        print(self.end_points)

#########################################################################################################################################
class SSD_model_mobilev2_fssd(SSD_model):
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)

        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layer10", "layer11", "layer12", "layer13", "layer14", "layer15"],
                                    feat_shapes=[(38, 38),
                                                 (19, 19),
                                                 (10, 10),
                                                 (5, 5),
                                                 (3, 3),
                                                 (1, 1)],
                                    anchor_size_bounds=[0.10, 0.90],  # diff from the original paper
                                    anchor_sizes=[(30., 78.),
                                                  (78., 126.),
                                                  (126., 174.),
                                                  (174., 222.),
                                                  (222., 270.),
                                                  (270., 318.)],
                                    anchor_ratios=[[2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],
                                                   [2, .5],
                                                   [2, .5]],
                                    anchor_steps=[8, 16, 30, 60, 100, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model_mobilev2_fssd.serving_input_receiver_fn)
    
    def _backbone(self, x, is_training):
        exp = 6
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3])
        
        # mobilev2
        ####### 1. 2D卷积块  =  2D卷积层 + BN + RELU6  3*3*3*32 步长2 32个通道输出 ############
        net = conv_bn_relu(net, 32, 3, 2, is_training, scope="conv1_1") # 150
        self.end_points["layer1"] = net
        ####### 2. 1个自适应残差深度可拆解模块  中间层扩张倍数为1为32*1  16个通道输出 ############
        net = res_block(net, 1, 16, 1, is_training, shortcut=False, scope='res2_1')
        self.end_points["layer2"] = net
        ####### 3. 2个自适应残差深度可拆解模块  中间层扩张倍数为6为16*6  24个通道输出 ############
        net = res_block(net, exp, 24, 2, is_training, scope='res3_1')  #  步长2 size/4  75
        net = res_block(net, exp, 24, 1, is_training, scope='res3_2')
        self.end_points["layer3"] = net
        ####### 4. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为24*6  32个通道输出 ############
        net = res_block(net, exp, 32, 2, is_training, scope='res4_1')  # 步长2 size/8  38
        net = res_block(net, exp, 32, 1, is_training, scope='res4_2')
        net = res_block(net, exp, 32, 1, is_training, scope='res4_3')
        self.end_points["layer4"] = net
        ####### 5. 4个自适应残差深度可拆解模块  中间层扩张倍数为6为32*6  64个通道输出 ############
        net = res_block(pad2d(net, 2), exp, 64, 2, is_training, scope='res5_1', padding="valid")# 步长2 size/16  20
        net = res_block(net, exp, 64, 1, is_training, scope='res5_2')
        net = res_block(net, exp, 64, 1, is_training, scope='res5_3')
        net = res_block(net, exp, 64, 1, is_training, scope='res5_4')
        self.end_points["layer5"] = net
        ####### 6. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为64*6  96个通道输出 ############
        net = res_block(net, exp, 96, 1, is_training, scope='res6_1')
        net = res_block(net, exp, 96, 1, is_training, scope='res6_2')
        net = res_block(net, exp, 96, 1, is_training, scope='res6_3')
        self.end_points["layer6"] = net
        ####### 7. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为96*6  160个通道输出 ###########
        net = res_block(net, exp, 160, 2, is_training, scope='res7_1')  # 步长2 size/32 10
        net = res_block(net, exp, 160, 1, is_training, scope='res7_2')
        net = res_block(net, exp, 160, 1, is_training, scope='res7_3')
        self.end_points["layer7"] = net
        ####### 8. 1个自适应残差深度可拆解模块  中间层扩张倍数为6为160*6  320个通道输出 ##########
        net = res_block(net, exp, 320, 1, is_training, shortcut=False, scope='res8_1')# 不进行残差合并 f(x)
        self.end_points["layer8"] = net
        
        return net
        
    def _extra_layer(self, net, is_training):
        net = tf.concat([upsample_conv2d(pad2d(self.end_points["layer4"], 1), 256, upsampling=False, scope='upsample9_1'),
                         upsample_conv2d(self.end_points["layer6"], 256, (40,40), scope='upsample9_2'),
                         upsample_conv2d(self.end_points["layer8"], 256, (40,40), scope='upsample9_3')], -1)
        
        net = batch_norm(net, training=is_training, scope="batch_norm9_4")
        self.end_points["layer9"] = net
        
        # additional SSD layers        
        net = ssd_conv2d(net, 512, 1, is_training, scope="ssd_conv2d10_1") # 38*38
        self.end_points["layer10"] = net
        net = ssd_conv2d(pad2d(net, 1), 512, 2, is_training, scope="ssd_conv2d11_1") # 19*19
        self.end_points["layer11"] = net
        
        net = ssd_conv2d(pad2d(net, 1), 256, 2, is_training, scope="ssd_conv2d12_1") # 10*10
        self.end_points["layer12"] = net
        net = ssd_conv2d(pad2d(net, 1), 256, 2, is_training, scope="ssd_conv2d13_1") # 5*5
        self.end_points["layer13"] = net
        net = ssd_conv2d(net, 256, 1, is_training, scope="ssd_conv2d14_1") # 3*3
        self.end_points["layer14"] = net
        net = ssd_conv2d(net, 256, 1, is_training, scope="ssd_conv2d15_1") # 1*1
        self.end_points["layer15"] = net

#########################################################################################################################################
class SSD_model_mobilev2_fpn(SSD_model):
    def __init__(self, model_path):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_path = model_path
        self.ssd_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                    model_dir=self.model_path)

        self.threshold = 0.5  # class score threshold
        self.ssd_params = SSDParams(img_shape=(300, 300),
                                    num_classes=2,
                                    no_annotation_label=21,
                                    feat_layers=["layerp4", "layerp8", "layerp9", "layerp10", "layerp11", "layer12"],
                                    feat_shapes=[(37, 37),
                                                 (18, 18),
                                                 (9, 9),
                                                 (4, 4),
                                                 (2, 2),
                                                 (1, 1)],
                                    anchor_size_bounds=[0.05, 0.95],  # diff from the original paper
                                    anchor_sizes=[(15., 69.),
                                                  (69., 123.),
                                                  (123., 177.),
                                                  (177., 231.),
                                                  (231., 285.),
                                                  (285., 339.)],
                                    anchor_ratios=[[.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75],
                                                   [.5, 0.75]],
                                    anchor_steps=[9, 17, 34, 75, 150, 300],
                                    anchor_offset=0.5,
                                    normalizations=[-1, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2])
        
        self.anchor_bboxes_list = ssd_anchors_all_layers(self.ssd_params.img_shape,
                                                         self.ssd_params.feat_shapes,
                                                         self.ssd_params.anchor_sizes,
                                                         self.ssd_params.anchor_ratios,
                                                         self.ssd_params.anchor_steps,
                                                         self.ssd_params.anchor_offset,
                                                         np.float32)

        self.predictor = tf.contrib.predictor.from_estimator(self.ssd_estimator, SSD_model_mobilev2_fpn.serving_input_receiver_fn)
    
    def _backbone(self, x, is_training):
        exp = 6
        
        # Input Layer
        net = tf.reshape(x, [-1, 300, 300, 3])
        
        # mobilev2
        ####### 1. 2D卷积块  =  2D卷积层 + BN + RELU6  3*3*3*32 步长2 32个通道输出 ############
        net = conv_bn_relu(pad2d(net, 1), 32, 3, 2, is_training, scope="conv1_1", padding="valid") # 150
        self.end_points["layer1"] = net
        ####### 2. 1个自适应残差深度可拆解模块  中间层扩张倍数为1为32*1  16个通道输出 ############
        net = res_block(net, 1, 16, 1, is_training, shortcut=False, scope='res2_1')
        self.end_points["layer2"] = net
        ####### 3. 2个自适应残差深度可拆解模块  中间层扩张倍数为6为16*6  24个通道输出 ############
        net = res_block(pad2d(net, 1), exp, 24, 2, is_training, scope='res3_1', padding="valid")  #  步长2 size/4  75
        net = res_block(net, exp, 24, 1, is_training, scope='res3_2')
        self.end_points["layer3"] = net
        ####### 4. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为24*6  32个通道输出 ############
        net = res_block(net, exp, 32, 2, is_training, scope='res4_1', padding="valid")  # 步长2 size/8  37
        net = res_block(net, exp, 32, 1, is_training, scope='res4_2')
        net = res_block(net, exp, 32, 1, is_training, scope='res4_3')
        self.end_points["layer4"] = net
        ####### 5. 4个自适应残差深度可拆解模块  中间层扩张倍数为6为32*6  64个通道输出 ############
        net = res_block(net, exp, 64, 2, is_training, scope='res5_1', padding="valid")# 步长2 size/16  18
        net = res_block(net, exp, 64, 1, is_training, scope='res5_2')
        net = res_block(net, exp, 64, 1, is_training, scope='res5_3')
        net = res_block(net, exp, 64, 1, is_training, scope='res5_4')
        self.end_points["layer5"] = net
        ####### 6. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为64*6  96个通道输出 ############
        net = res_block(net, exp, 96, 1, is_training, scope='res6_1')
        net = res_block(net, exp, 96, 1, is_training, scope='res6_2')
        net = res_block(net, exp, 96, 1, is_training, scope='res6_3')
        self.end_points["layer6"] = net
        ####### 7. 3个自适应残差深度可拆解模块  中间层扩张倍数为6为96*6  160个通道输出 ###########
        net = res_block(net, exp, 160, 1, is_training, scope='res7_1')  # 步长2 size/32
        net = res_block(net, exp, 160, 1, is_training, scope='res7_2')
        net = res_block(net, exp, 160, 1, is_training, scope='res7_3')
        self.end_points["layer7"] = net
        ####### 8. 1个自适应残差深度可拆解模块  中间层扩张倍数为6为160*6  320个通道输出 ##########
        net = res_block(net, exp, 320, 1, is_training, shortcut=False, scope='res8_1')# 不进行残差合并 f(x)
        self.end_points["layer8"] = net
        
        return net
        
    def _extra_layer(self, net, is_training):
        # additional SSD layers
        net = ssd_conv2d(pad2d(net, 1), 512, 2, is_training, scope="ssd_conv2d12_1") # 9
        self.end_points["layer9"] = net
        net = ssd_conv2d(net, 256, 2, is_training, scope="ssd_conv2d13_1") # 4
        self.end_points["layer10"] = net
        net = ssd_conv2d(net, 256, 1, is_training, scope="ssd_conv2d14_1") # 2
        self.end_points["layer11"] = net
        net = ssd_conv2d(pad2d_single(net, 1), 256, 1, is_training, scope="ssd_conv2d15_1") # 1
        self.end_points["layer12"] = net
        
        net = conv2d(net, 256, 1, scope="convp12_1")
        net = fpn_block(self.end_points["layer11"], net, 256, is_training, scope="layerp11") # 2
        self.end_points["layerp11"] = net
        net = fpn_block(self.end_points["layer10"], net, 256, is_training, scope="layerp10") # 4
        self.end_points["layerp10"] = net
        net = fpn_block(self.end_points["layer9"], net, 256, is_training, padding=True, scope="layerp9") # 9
        self.end_points["layerp9"] = net
        net = fpn_block(self.end_points["layer8"], net, 256, is_training, scope="layerp8") # 18
        self.end_points["layerp8"] = net
        net = fpn_block(self.end_points["layer4"], net, 256, is_training, padding=True, scope="layerp4") # 37
        self.end_points["layerp4"] = net