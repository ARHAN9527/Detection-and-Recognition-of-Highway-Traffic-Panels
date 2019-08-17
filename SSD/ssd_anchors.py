"""
SSD anchors
"""
import math

import numpy as np

def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes

def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """
    Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)  # [size, size, 1]
    x = np.expand_dims(x, axis=-1)  # [size, size, 1]

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)  # [n_anchors]
    w = np.zeros((num_anchors, ), dtype=dtype)  # [n_anchors]
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)

    return y, x, h, w

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers."""
    layers_anchors = []
    for i, feat_shape in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape,
                                             feat_shape,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset,
                                             dtype=dtype)
        layers_anchors.append(anchor_bboxes)

    return layers_anchors

if __name__ == "__main__":
    from collections import namedtuple
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

    ssd_params = SSDParams(img_shape=(300, 300),
                           num_classes=21,
                           no_annotation_label=21,
                           feat_layers=["block4", "block7", "block8", "block9", "block10", "block11"],
                           feat_shapes=[(38, 38),
                                        (19, 19),
                                        (10, 10),
                                        (5, 5),
                                        (3, 3),
                                        (1, 1)],
                           anchor_size_bounds=[0.15, 0.90],  # diff from the original paper
                           anchor_sizes=[(21., 45.),
                                         (45., 99.),
                                         (99., 153.),
                                         (153., 207.),
                                         (207., 261.),
                                         (261., 315.)],
                           anchor_ratios=[[2, .5],
                                          [2, .5, 3, 1. / 3],
                                          [2, .5, 3, 1. / 3],
                                          [2, .5, 3, 1. / 3],
                                          [2, .5],
                                          [2, .5]],
                           anchor_steps=[8, 16, 32, 64, 100, 300],
                           anchor_offset=0.5,
                           normalizations=[20, -1, -1, -1, -1, -1],
                           prior_scaling=[0.1, 0.1, 0.2, 0.2])

    anchor_bboxes_list =  ssd_anchors_all_layers(ssd_params.img_shape,
                                                 ssd_params.feat_shapes,
                                                 ssd_params.anchor_sizes,
                                                 ssd_params.anchor_ratios,
                                                 ssd_params.anchor_steps,
                                                 ssd_params.anchor_offset,
                                                 np.float32)

    print((anchor_bboxes_list[0][2].shape))