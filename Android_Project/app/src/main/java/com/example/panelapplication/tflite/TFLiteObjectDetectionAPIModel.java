package com.example.panelapplication.tflite;

import android.app.Application;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Environment;
import android.os.Trace;

import org.tensorflow.lite.Interpreter;

import com.example.panelapplication.Postprocessing;
import com.example.panelapplication.env.Logger;

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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
    // Float model
    //private static final float IMAGE_MEAN = 128.0f;
    //private static final float IMAGE_STD = 128.0f;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    //private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    //private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    //private float[][] outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    //private float[] numDetections;

    // Only return this many results.
    private static final int NUM_DETECTIONS = 7760; // 7760 2000
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
    private float[][] tfoutput_detect;

    private Postprocessing postPro;
    private ArrayList<float[]> nmsBoxes;

    private int frameCount = 0;
    private long totalTime = 0;
    private long startTime = 0;

    private TFLiteObjectDetectionAPIModel() {}

    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

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
        final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

        d.labels = readLabelFile(assetManager, labelFilename);
        d.inputSize = inputSize;

        // NEW: Prepare GPU delegate.
        //GpuDelegate delegate = new org.tensorflow.lite.Delegate();
        //Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);

        try {
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), modelFilename);
            d.tfLite = new Interpreter(file);
            //d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
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
        d.tfoutput_detect = new float[NUM_DETECTIONS][5];
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
    public String recognizeImage(final Bitmap bitmap) {
        String label = null;
        return label;
    }

    @Override
    public List<Recognition> detectImage(final Bitmap bitmap) {
        //long startTime = System.currentTimeMillis();

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmapToInputData(bitmap);
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        /*outputLocations = new float[1][NUM_DETECTIONS][4];
        outputClasses = new float[1][NUM_DETECTIONS];
        outputScores = new float[1][NUM_DETECTIONS];
        numDetections = new float[1];*/

        /*Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();*/

        /////////////////////////////////////////////
        //float [][]tfoutput = new float[7760][5]; //2134
        ////////////////////////////////////////////

        /*outputMap.put(0, outputClasses);
        outputMap.put(1, outputScores);
        outputMap.put(2, outputLocations);
        outputMap.put(3, numDetections);*/
        Trace.endSection();

        //startTime = System.currentTimeMillis();

        // Run the inference call.
        Trace.beginSection("run");
        //tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        tfLite.run(imgData, tfoutput_detect);
        Trace.endSection();

        //totalTime += (System.currentTimeMillis()-startTime);
        //frameCount++;
        //LOGGER.w("average time= "+(totalTime)/frameCount);

        postPro = new Postprocessing(tfoutput_detect);
        nmsBoxes = postPro.postDetect();

        // Show the best detections.
        // after scaling them back to the input size.
        final ArrayList<Recognition> recognitions = new ArrayList<>();

        for (int i = 0; i < nmsBoxes.size(); ++i) {
            final RectF detection =
                    new RectF(nmsBoxes.get(i)[2],
                              nmsBoxes.get(i)[1],
                              nmsBoxes.get(i)[4],
                              nmsBoxes.get(i)[3]);
            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            int labelOffset = 1;
            recognitions.add(
                    new Recognition(
                            "" + i,
                            labels.get(0 + labelOffset),
                            nmsBoxes.get(i)[0],
                            detection));
        }

        /*
        final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
        for (int i = 0; i < NUM_DETECTIONS; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[0][i][1] * inputSize,
                            outputLocations[0][i][0] * inputSize,
                            outputLocations[0][i][3] * inputSize,
                            outputLocations[0][i][2] * inputSize);
            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            int labelOffset = 1;
            recognitions.add(
                    new Recognition(
                            "" + i,
                            labels.get((int) outputClasses[0][i] + labelOffset),
                            outputScores[0][i],
                            detection));
        }
        */

        Trace.endSection(); // "recognizeImage"

        //LOGGER.w(""+(System.currentTimeMillis()-startTime));

        return recognitions;
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