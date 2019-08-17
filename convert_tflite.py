import tensorflow as tf
from pathlib import Path
'''
from_saved_model(saved_model_dir,
                 input_arrays=None,
                 input_shapes=None,
                 output_arrays=None,
                 tag_set=None,
                 signature_key=None)

method of builtins.type instance
Creates a TFLiteConverter class from a SavedModel.
    Args:
        saved_model_dir:
            SavedModel directory to convert.

        input_arrays: (default None)
            List of input tensors to freeze graph with.
            Uses input arrays from SignatureDef when none are provided.

        input_shapes: (default None)
            Dict of strings representing input tensor names to list of integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
            Automatically determined when input shapes is None (e.g., {"foo" : None}).

        output_arrays: (default None)
            List of output tensors to freeze graph with.
            Uses output arrays from SignatureDef when none are provided.

        tag_set: (default set("serve"))
            Set of tags identifying the MetaGraphDef within the SavedModel to analyze.
            All tags in the tag set must be present.


        signature_key: (default DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            Key identifying SignatureDef containing inputs and outputs.

    Returns:
      TFLiteConverter class.
'''
model_dir = "saved_model_mobilev2_ssd_width0125_19"
saved_model_dir = "SSD/" + model_dir # SSD/saved_model_ssd_ratio335075 MNIST/saved_mnist_0527 saved_mnist_0603
save_file = ".tflite"

def get_pred_fn(dir_path):
    export_dir = dir_path
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1]) # SSD/saved_model/1556852589
    return latest

if 1:
    converter = tf.lite.TFLiteConverter.from_saved_model(get_pred_fn(saved_model_dir),
                                                         input_arrays=["x"],
                                                         input_shapes={"x" : [1, 300, 300, 3]},
                                                         output_arrays=["tfoutput"]) # [1, 300, 300, 3]

else:
    converter = tf.lite.TFLiteConverter.from_saved_model(get_pred_fn(saved_model_dir),
                                                         input_arrays=["x"],
                                                         input_shapes={"x" : [1, 128, 128, 3]},
                                                         output_arrays=["softmax_tensor"])

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.post_training_quantize = True
#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                        tf.lite.OpsSet.SELECT_TF_OPS]
#converter.allow_custom_ops=True
tflite_model = converter.convert()

#save_file = model_dir + save_file
save_file = get_pred_fn(saved_model_dir).split("\\")[-1] + save_file

open(save_file, "wb").write(tflite_model)