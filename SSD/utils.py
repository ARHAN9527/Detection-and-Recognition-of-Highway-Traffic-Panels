import random

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.29         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.29
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)

def _random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    """
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        #image = ops.convert_to_tensor(image, name='image')
        #_Check3DImage(image, require_static=False)
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        # Flip image.
        image = control_flow_ops.cond(mirror_cond,
                                       lambda: array_ops.reverse_v2(image, [1]),
                                       lambda: image)
        # Flip bboxes.
        bboxes = control_flow_ops.cond(mirror_cond,
                                       lambda: flip_bboxes(bboxes),
                                       lambda: bboxes)
        return image, bboxes

def _resize_image(image, 
                  size,
                  method=tf.image.ResizeMethod.BILINEAR,
                  align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, 
                                       size,
                                       method, 
                                       align_corners)
        image = tf.reshape(image,(size[0], size[1], 3))
        return image


def _safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(math_ops.greater(denominator, 0),
                    math_ops.divide(numerator, denominator),
                    tf.zeros_like(numerator),
                    name=name)

def bboxes_intersection(bbox_ref, bboxes, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = _safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores

def _bboxes_filter_overlap(labels, 
                           bboxes,
                           threshold=0.5, 
                           assign_negative=False,
                           scope=None):
    """Filter out bounding boxes based on (relative )overlap with reference
    box [0, 0, 1, 1].  Remove completely bounding boxes, or assign negative
    labels to the one outside (useful for latter processing...).

    Return:
      labels, bboxes: Filtered (or newly assigned) elements.
    """
    
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        mask = scores > threshold
        if assign_negative:
            labels = tf.where(mask, labels, -labels)
            # bboxes = tf.where(mask, bboxes, bboxes)
        else:
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
            
        return labels, bboxes

def _bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(tf.shape(image),
                                                                                     bounding_boxes=tf.expand_dims(bboxes, 0),
                                                                                     min_object_covered=min_object_covered,
                                                                                     aspect_ratio_range=aspect_ratio_range,
                                                                                     area_range=area_range,
                                                                                     max_attempts=max_attempts,
                                                                                     use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0] # 1,1,4

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = _bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = _bboxes_filter_overlap(labels, 
                                                bboxes,
                                                threshold=0.8,
                                                assign_negative=False)
        
        return cropped_image, labels, bboxes, distort_bbox

def process_img(image,
                labels,
                bboxes,
                out_shape=EVAL_SIZE,
                min_object_covered=MIN_OBJECT_COVERED,
                aspect_ratio_range=CROP_RATIO_RANGE):
    
    tf.reset_default_graph() #
    with tf.Session() as sess:
        timg = tf.convert_to_tensor(image, dtype=tf.float32)

        # Random distort bounding box crop
        crop_dst_image, crop_labels, crop_bboxes, distort_bbox = distorted_bounding_box_crop(timg, 
                                                                                             labels[0], 
                                                                                             bboxes[0],
                                                                                             min_object_covered=min_object_covered,
                                                                                             aspect_ratio_range=aspect_ratio_range)

        # Resize image
        resize_dst_image = _resize_image(image=crop_dst_image, 
                                         size=out_shape,
                                         method=tf.image.ResizeMethod.BILINEAR,
                                         align_corners=False)

        # random distort color
        '''disort_image = apply_with_random_selector(resize_dst_image,
                                                  lambda x, ordering: distort_color(x, ordering, fast_mode),
                                                  num_cases=4)'''

        # Random flip horizontal
        flip_dst_image, flip_bboxes = _random_flip_left_right(resize_dst_image, crop_bboxes)
        
        # Limit bbox range (0,1)
        flip_bboxes = tf.transpose(flip_bboxes)
        b0 = tf.expand_dims(tf.maximum(flip_bboxes[0], 0), axis=0)
        b1 = tf.expand_dims(tf.maximum(flip_bboxes[1], 0), axis=0)
        b2 = tf.expand_dims(tf.minimum(flip_bboxes[2], 1), axis=0)
        b3 = tf.expand_dims(tf.minimum(flip_bboxes[3], 1), axis=0)
        limit_bboxes = tf.transpose(tf.concat((b0, b1, b2, b3), axis=0))
        
        cropped_image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(flip_dst_image, 0), 
                                                              tf.expand_dims(limit_bboxes, 0))

        #stan_image = tf.image.per_image_standardization(flip_dst_image)
        
        out_dst_image, out_labels, out_bboxes, out_cropped_image_with_box = sess.run([tf.expand_dims(flip_dst_image, 0), 
                                                                                      tf.expand_dims(crop_labels, 0), 
                                                                                      tf.expand_dims(limit_bboxes, 0),
                                                                                      cropped_image_with_box[0]])
    tf.get_default_graph().finalize() #
    
    return out_dst_image, np.array(out_labels, np.int64), out_bboxes, out_cropped_image_with_box

############## preprocess label ##################

def preprocess_label(bbox, img_shape):
    """Preprocess a image to inference"""
    cbbox = np.array(bbox, np.float32)
    trbbox = np.transpose(cbbox)
    trbbox[0] /= img_shape[0]-1
    trbbox[1] /= img_shape[1]-1
    trbbox[2] /= img_shape[0]-1
    trbbox[3] /= img_shape[1]-1
    rbbox = np.transpose(trbbox)
    return rbbox

############## preprocess image ##################
def stan_image(image):
    """Standardize the value range of an array."""
    if image.ndim != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
        
    with tf.Session() as sess:
        img_stan = sess.run(tf.image.per_image_standardization(image))    
    
    return img_stan

def norm_image(image):
    """Normalizes the value range of an array."""
    if image.ndim != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    return cv2.normalize(image, None, 0, 1, norm_type=cv2.NORM_MINMAX)

# whiten the image
def whiten_image(image, means=(123., 117., 104.)):
    """Subtracts the given means from each image channel"""
    if image.ndim != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = np.array(means, dtype=image.dtype)
    image = image - mean
    return image

def resize_image(image, size=(300, 300)):
    return cv2.resize(image, size)

def preprocess_image(image, 
                     labels=None,
                     bboxes=None,
                     out_shape=EVAL_SIZE,
                     min_object_covered=MIN_OBJECT_COVERED,
                     aspect_ratio_range=CROP_RATIO_RANGE,
                     num=0):
    """Preprocess a image to inference"""
    if num == 0:
        num = random.randint(1,2)
    
    if num == 1:
        image_cp = np.copy(image).astype(np.float32)
        # whiten the image
        #image_whitened = whiten_image(image_cp)
        # standardize the image
        #image_stan = stan_image(image_cp)
        # normalize the image
        #image_norm = norm_image(image_cp)
        # resize the image
        image_resized = resize_image(image_cp, out_shape)
        # expand the batch_size dim
        img_prepocessed = np.expand_dims(image_resized, axis=0)
        
        if labels == None and bboxes == None:
            return img_prepocessed

        label_prepocessed = np.array(labels, np.int64)
        box_prepocessed = np.array(bboxes)        
        return img_prepocessed, label_prepocessed, box_prepocessed, None

    else:
        return  process_img(image=image,
                            labels=labels,
                            bboxes=bboxes,
                            out_shape=out_shape,
                            min_object_covered=min_object_covered,
                            aspect_ratio_range=aspect_ratio_range)
    
############## process bboxes ##################
def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

def bboxes_iou(bboxes1, bboxes2):
    """Computing iou between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    iou = int_vol / (vol1 + vol2 - int_vol)
    return iou

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5, score_threshold=0.5):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(np.logical_and(keep_bboxes, scores > score_threshold))
    return classes[idxes], scores[idxes], bboxes[idxes]

def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes
def process_tflite(tflite):    
    rscores = tflite[:, 0]
    rbboxes = tflite[:, 1:]
    rclasses = np.ones_like(rscores, dtype=np.int8)
    
    return rclasses, rscores, rbboxes

def process_bboxes(rclasses=None,
                   rscores=None,
                   rbboxes=None,
                   tflite=None,
                   rbbox_img = (0.0, 0.0, 1.0, 1.0),
                   top_k=200,
                   nms_threshold=0.5):
    """Process the bboxes including sort and nms"""
    if (rclasses==None and rscores==None and rbboxes==None):
        rclasses, rscores, rbboxes = process_tflite(tflite)
    
    rbboxes = bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = bboxes_sort(rclasses, rscores, rbboxes, top_k)
    rclasses, rscores, rbboxes = bboxes_nms(rclasses, rscores, rbboxes, nms_threshold)
    rbboxes = bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

############################process true positive###################################################

def process_tp(rclasses, rscores, rbboxes, gtclasses, gtboxes, iou_threshold=0.5):
    tp_boxes = np.zeros(rclasses.shape, dtype=np.bool)
    for i in range(gtclasses.shape[0]):
        iou_max = 0
        iou_max_idx = -1
        for j in range(rbboxes.shape[0]):
            if tp_boxes[j]:
                continue
            iou = bboxes_iou(gtboxes[i], rbboxes[j])
            if iou > iou_threshold and iou > iou_max:
                iou_max = iou
                iou_max_idx = j
        
        if iou_max_idx != -1:
            tp_boxes[iou_max_idx] = True
            
    return tp_boxes.tolist()       