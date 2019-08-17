package com.example.panelapplication.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Environment;
import android.os.Trace;

import com.example.panelapplication.Postprocessing;
import com.example.panelapplication.env.Logger;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */

public class TFLiteRecognitionAPIModel implements Classifier {
    // Only return this many results.
    private static final int NUM_CLASSES = 42;
    private static final Logger LOGGER = new Logger();
    private Interpreter tfLite;
    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();

    private boolean isModelQuantized;
    // Config values.
    private int inputSize;
    private int[] intValues;
    // tf input data
    private ByteBuffer imgData;
    // tf output data
    private float[][] tfoutput_recognize;

    private Postprocessing postPro;
    private int predictClass;

    private int frameCount = 0;
    private long totalTime = 0;
    private long startTime = 0;

    private TFLiteRecognitionAPIModel() {}

    private static Vector<String> readLabelFile(final AssetManager assetManager,
                                                final String labelFilename) throws IOException {
        Vector<String> labels = new Vector<String>();
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            labels.add(line);
        }
        br.close();

        return labels;
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize The size of image input
     * @param isQuantized Boolean representing model is quantized or not
     */
    public static Classifier create(final AssetManager assetManager,
                                    final String modelFilename,
                                    final String labelFilename,
                                    final int inputSize,
                                    final boolean isQuantized) throws IOException {
        final TFLiteRecognitionAPIModel d = new TFLiteRecognitionAPIModel();

        d.labels = readLabelFile(assetManager, labelFilename);
        d.inputSize = inputSize;

        // NEW: Prepare GPU delegate.
        //GpuDelegate delegate = new org.tensorflow.lite.Delegate();
        //Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);

        try {
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), modelFilename);
            d.tfLite = new Interpreter(file);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (d.isModelQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel); ////////////////////// 3 -> 2
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.inputSize * d.inputSize];

        d.tfLite.setNumThreads(NUM_THREADS);
        d.tfoutput_recognize = new float[1][NUM_CLASSES];
        /*d.outputLocations = new float[1][NUM_DETECTIONS][4];
        d.outputClasses = new float[1][NUM_DETECTIONS];
        d.outputScores = new float[1][NUM_DETECTIONS];
        d.numDetections = new float[1];*/
        return d;
    }

    private void bitmapToInputData(Bitmap bitmap) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                    /*
                    imgData.put((byte) ((pixelValue >> 11) & 0x1F));
                    imgData.put((byte) ((pixelValue >> 5) & 0x3F));
                    imgData.put((byte) (pixelValue & 0x1F));
                    */
                } else { // Float model
                    imgData.putFloat((pixelValue >> 16) & 0xFF);
                    imgData.putFloat((pixelValue >> 8) & 0xFF);
                    imgData.putFloat(pixelValue & 0xFF);
                    /*
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    */


                }
            }
        }
    }

    @Override
    public List<Recognition> detectImage(final Bitmap bitmap) {
        List<Recognition> recognitions = null;
        return recognitions;
    }

    @Override
    public String recognizeImage(final Bitmap bitmap) {
        //long startTime = System.currentTimeMillis();

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmapToInputData(bitmap);
        Trace.endSection(); // preprocessBitmap

        // Run the inference call.
        Trace.beginSection("run");

        tfLite.run(imgData, tfoutput_recognize);

        Trace.endSection();
        postPro = new Postprocessing(tfoutput_recognize);
        predictClass = postPro.postRecognize();
        Trace.endSection(); // "recognizeImage"

        //LOGGER.w(""+(System.currentTimeMillis()-startTime));
        return labels.get(predictClass);
    }

    @Override
    public void enableStatLogging(final boolean logStats) {}

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {}

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }

}