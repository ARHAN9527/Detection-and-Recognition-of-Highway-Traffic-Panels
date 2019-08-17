package com.example.panelapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.os.Environment;
import android.os.Handler;
import android.speech.tts.TextToSpeech;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import com.example.panelapplication.env.ImageUtils;
import com.example.panelapplication.env.Logger;
import com.example.panelapplication.tflite.Classifier;

import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Locale;

public class VideoSurfaceView extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final Logger LOGGER = new Logger();

    private static final int COLOR_FormatI420 = 1;
    private static final int COLOR_FormatNV21 = 2;
    private static final int FILE_TypeI420 = 1;
    private static final int FILE_TypeNV21 = 2;
    private static final int FILE_TypeJPEG = 3;

    private final boolean VERBOSE = false;
    private boolean isRunning = false;

    private AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
    private OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
    private OpenCVFrameConverter.ToOrgOpenCvCoreMat converterToOrgMat = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();

    private SurfaceHolder mSurfaceHolder;
    private Canvas mCanvas; // 绘图的Canvas
    private MediaExtractor videoExtractor = new MediaExtractor();
    private MediaFormat mediaFormat = null;
    private MediaCodec mVideoCodec = null; //参数为MediaFormat类中的MIMETYPE
    private String mime = null;
    private int previewWidth = 1920+160;
    private int previewHeight = 1080+90;
    private Rect rect = new Rect(0,0, previewWidth, previewHeight);
    private Bitmap detectBitmap = null;
    private Bitmap recognizeBitmap = null;
    private Mat mMatRgb;

    private Classifier detector;
    private Classifier recognition;
    private List<Classifier.Recognition> results;
    private RectF location;
    private String label;
    private int cropSize = 300;
    private int panelSize = 128;
    private Matrix cropToFrameTransform = ImageUtils.getTransformationMatrix(1,
                                                                          1,
                                                                          previewWidth,
                                                                          previewHeight,
                                                                          0,
                                                                          false);

    private final Paint textPaint = new Paint();
    private final Paint rectPaint = new Paint();

    public String videoFile = null;
    private long startTime = 0;
    private long delayTime = 0;
    private long spf;
    private final DecimalFormat formatter = new DecimalFormat("#.###");

    private long totalTime = 0;
    private long frameCount = 0;
    private long frameTime = 0;
    private int frameSkip = 0;
    private Handler handler = new Handler();
    private Runnable runnable = new Runnable(){

        @Override
        public void run() { ;
        }};

    public TextToSpeech tts;
    private float RightPanelLocation;
    private String RightPanelLabel = "";
    private String LastRightPanelLabel = "";

    public void setVideoFile(String fileName) {
        this.videoFile = fileName;
    }

    public void setIsRunning(boolean isRunning){
        this.isRunning = isRunning;
    }

    public void setDetector(Classifier detector) {
        this.detector = detector;
    }

    public void setRecognition(Classifier recognition) {
        this.recognition = recognition;
    }

    public VideoSurfaceView(Context context) {
        this(context, null);
    }

    public VideoSurfaceView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public VideoSurfaceView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        initView();
    }

    private void initView(){
        /* 初始化View */
        mSurfaceHolder = getHolder();
        mSurfaceHolder.addCallback(this);
        setFocusable(true);
        setKeepScreenOn(true);
        setFocusableInTouchMode(true);
    }

    private void delayCorrect(long spendTime) {
        frameSkip = (int)(spendTime/spf)+1;
        long remainder = (spendTime%spf);
        delayTime = spf - remainder;
        if (remainder > 0){
            handler.postDelayed(runnable, delayTime);

            /*try {
                Thread.sleep(delayTime);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (IllegalArgumentException e) {
            }*/
        } else {
            frameSkip--;
        }

    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(40.0f);

        rectPaint.setColor(Color.RED);
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(1.0f);

        //final int decodeColorFormat = MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible;
        File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), videoFile);
        logDebug("path = "+file.getAbsolutePath());
        try {
            videoExtractor.setDataSource(file.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
            logDebug("setDataSource FAIL");
        }

        int videoTrackIndex = -1;
        videoTrackIndex = getTrackIndex();
        if (videoTrackIndex == -1) {
            logDebug("video track is not found.");
            return;
        }

        videoExtractor.selectTrack(videoTrackIndex);
        mediaFormat = videoExtractor.getTrackFormat(videoTrackIndex); //根据视轨id获得对应的MediaForamt

        int width = mediaFormat.getInteger(MediaFormat.KEY_WIDTH);
        int height = mediaFormat.getInteger(MediaFormat.KEY_HEIGHT);
        long time = mediaFormat.getLong(MediaFormat.KEY_DURATION);
        spf = (long)(1.0 / mediaFormat.getInteger(MediaFormat.KEY_FRAME_RATE)*1000);

        try {
            mVideoCodec = MediaCodec.createDecoderByType(mediaFormat.getString(MediaFormat.KEY_MIME)); // create MediaCodec by type
        } catch (IOException e) {
            e.printStackTrace();
        }

        showSupportedColorFormat(mVideoCodec.getCodecInfo().getCapabilitiesForType(mime));

        /*
        if (isColorFormatSupported(decodeColorFormat, mVideoCodec.getCodecInfo().getCapabilitiesForType(mime))) {
            //mediaFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, decodeColorFormat);
            logDebug("set decode color format to type " + decodeColorFormat);
        } else {
            logDebug("unable to set decode color format, color format type " + decodeColorFormat + " not supported");
        }
        */

        mVideoCodec.setCallback(new MediaCodec.Callback() {
            @Override
            public void onInputBufferAvailable(MediaCodec codec, int inputBufferId) {
                ByteBuffer inputBuffer = codec.getInputBuffer(inputBufferId);
                // fill inputBuffer with valid data
                int sampleSize = videoExtractor.readSampleData(inputBuffer, 0);
                if (sampleSize < 0) {
                    codec.queueInputBuffer(inputBufferId, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                } else {
                    codec.queueInputBuffer(inputBufferId, 0, sampleSize, videoExtractor.getSampleTime(), 0);
                    videoExtractor.advance();
                }
            }

            @Override
            public void onOutputBufferAvailable(MediaCodec codec, int outputBufferId, MediaCodec.BufferInfo info) {

                if (outputBufferId >= 0) {
                    if (frameSkip > 0) {
                        frameSkip--;
                        frameCount++;
                        boolean doRender = (info.size != 0);
                        if (doRender) {
                            Image image = codec.getOutputImage(outputBufferId);
                            image.close();
                            codec.releaseOutputBuffer(outputBufferId, false);
                        }
                    } else {
                        startTime = System.currentTimeMillis();
                        boolean doRender = (info.size != 0);
                        int outputImageFileType = 0;
                        if (doRender) {
                            Image image = codec.getOutputImage(outputBufferId);

                            if (outputImageFileType != -1) {
                                drawSomething(image);
                                /*
                                switch (1) {
                                    case FILE_TypeI420:
                                        //drawSomething(getDataFromImage(image, COLOR_FormatI420));
                                        drawSomething(image);
                                        break;
                                    case FILE_TypeNV21:
                                        break;
                                    case FILE_TypeJPEG:
                                        break;
                                }
                                */
                            }
                            image.close();
                            codec.releaseOutputBuffer(outputBufferId, false);
                        }

                        frameTime = (System.currentTimeMillis() - startTime);

                        //totalTime += frameTime;
                        //frameCount++;
                        //LOGGER.w("average time= " + (totalTime / frameCount) + " frameTime="+frameTime);

                        delayCorrect(frameTime);
                    }
                    /*
                    try {
                        Thread.sleep(delayTime);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } catch (IllegalArgumentException e) {
                    }*/
                }
            }

            @Override
            public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {
                // Subsequent data will conform to new format.
                // Can ignore if using getOutputFormat(outputBufferId)
                mediaFormat = format; // option B
            }

            @Override
            public void onError(MediaCodec codec, MediaCodec.CodecException e) {
                e.printStackTrace();
            }
        });

        // 第一个参数是待解码的数据格式(也可用于编码操作);
        // 第二个参数是设置surface，用来在其上绘制解码器解码出的数据；
        // 第三个参数于数据加密有关；
        // 第四个参数上1表示编码器，0是否表示解码器呢？？
        //mVideoCodec.configure(mediaFormat, null, null, 0);
        //当configure好后，就可以调用start()方法来请求向MediaCodec的inputBuffer中写入数据了
        //mVideoCodec.start();
        //开启子线程
        new Thread(this).start();
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {

    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        if (mVideoCodec != null) {
            mVideoCodec.stop();
            mVideoCodec.release();
            mVideoCodec = null;
        }
        if (videoExtractor != null) {
            videoExtractor.release();
            videoExtractor = null;
        }
    }

    @Override
    public void run() {
        // 第一个参数是待解码的数据格式(也可用于编码操作);
        // 第二个参数是设置surface，用来在其上绘制解码器解码出的数据；
        // 第三个参数于数据加密有关；
        // 第四个参数上1表示编码器，0是否表示解码器呢？？
        while (true) {
            if (isRunning) {
                mVideoCodec.configure(mediaFormat, null, null, 0);
                //当configure好后，就可以调用start()方法来请求向MediaCodec的inputBuffer中写入数据了
                mVideoCodec.start();
                break;
            }
        }
    }
    //绘图逻辑
    private void drawSomething(Image image) {
        try {

            //Mat mat = new Mat(imgHeight, imgWidth, CvType.CV_8UC3);
            //Imgcodecs.imencode(".jpg", mat, new MatOfByte(bytes));
            //mat.put(0, 0, data);

            //mat = converterToOrgMat.convert(frame);
            //Mat mat2= converterToOrgMat.convert(frame);
            //Imgproc.resize(mat, mat2, new Size(300,300));


            //Frame frame = converterToOrgMat.convert(mat);

            //bitmap = converterToBitmap.convert(frame);// frame -> bitmap

            imageToMat(image);
            matToBitmap();

            //final List<Classifier.Recognition> results = detector.recognizeImage(bitmap);

            //startTime = System.currentTimeMillis();
            results = detector.detectImage(detectBitmap);
            RightPanelLocation = -1;
            //LOGGER.w(""+(System.currentTimeMillis()-startTime));
            //totalTime += (System.currentTimeMillis()-startTime);
            //frameCount++;
            //LOGGER.w("Average time= "+(totalTime/frameCount));


            //rect.set(0, 0, 1900, 1080);
            mCanvas = mSurfaceHolder.lockHardwareCanvas(); // 获得canvas对象 lockCanvas lockHardwareCanvas
            mCanvas.drawBitmap(recognizeBitmap, null, rect, null); // 绘图 bitmap.getWidth() bitmap.getHeight()
            //mCanvas.drawBitmap(bitmap, 0, 0, null);

            LastRightPanelLabel = RightPanelLabel;
            RightPanelLabel = "";
            for (final Classifier.Recognition result : results) {
                location = result.getLocation();

                /*
                cropToFrameTransform.mapRect(location);
                mCanvas.drawRect(location, rectPaint);
                mCanvas.drawText(result.getTitle()+ ":" + formatter.format(result.getConfidence()), location.left, location.bottom+40, textPaint);
                */

                label = recognition.recognizeImage(cropPanelbyLocation(location));

                if (!label.equals("0"))
                {
                    cropToFrameTransform.mapRect(location);

                    mCanvas.drawRect(location, rectPaint);
                    mCanvas.drawText(label+"", location.left, location.bottom+40, textPaint);
                    if (location.left > RightPanelLocation){
                        RightPanelLocation = location.left;
                        RightPanelLabel = label;
                    }
                }

                //mCanvas.drawText(result.getTitle()+ ":" + formatter.format(result.getConfidence()), location.left, location.bottom+40, textPaint);
                //cropToFrameTransform.mapRect(location);

                //result.setLocation(location);
                //mappedRecognitions.add(result);

            }
            if (!(RightPanelLabel.equals(LastRightPanelLabel)) && !(RightPanelLabel.equals(""))){
                tts.speak( RightPanelLabel, TextToSpeech.QUEUE_FLUSH, null, TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID);
            }



        } catch (Exception e){

        } finally {
            if (mCanvas != null){
                mSurfaceHolder.unlockCanvasAndPost(mCanvas); // 释放canvas对象并提交画布
            }
        }
    }

    private int getTrackIndex() {
        int numTracks=0;
        final String KEYMIME = "video/";

        numTracks = videoExtractor.getTrackCount();
        logDebug("numTracks = "+numTracks);

        for (int i = 0; i < numTracks; ++i) {
            mediaFormat = videoExtractor.getTrackFormat(i);
            mime = mediaFormat.getString(MediaFormat.KEY_MIME);
            logDebug("mime="+mime);

            if (mime.startsWith(KEYMIME)) {
                return i;
            }
        }
        return -1;
    }

    private void showSupportedColorFormat(MediaCodecInfo.CodecCapabilities caps) {
        logDebug("supported color format: ");
        for (int c : caps.colorFormats) {
            logDebug(c + "\t");
        }
    }

    private boolean isColorFormatSupported(int colorFormat, MediaCodecInfo.CodecCapabilities caps) {
        for (int c : caps.colorFormats) {
            if (c == colorFormat) {
                return true;
            }
        }
        return false;
    }

    private boolean isImageFormatSupported(Image image) {
        int format = image.getFormat();
        switch (format) {
            case ImageFormat.YUV_420_888:
            case ImageFormat.NV21:
            case ImageFormat.YV12:
                return true;
        }
        return false;
    }

    private byte[] getDataFromImage(Image image, int colorFormat) {
        if (colorFormat != COLOR_FormatI420 && colorFormat != COLOR_FormatNV21) {
            throw new IllegalArgumentException("only support COLOR_FormatI420 " + "and COLOR_FormatNV21");
        }
        if (!isImageFormatSupported(image)) {
            throw new RuntimeException("can't convert Image to byte array, format " + image.getFormat());
        }
        Rect crop = image.getCropRect();
        int format = image.getFormat();
        int width = crop.width();
        int height = crop.height();
        Image.Plane[] planes = image.getPlanes();
        logDebug("getBitsPerPixel="+ImageFormat.getBitsPerPixel(format));
        byte[] data = new byte[width * height * ImageFormat.getBitsPerPixel(format) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];
        logDebug("get data from " + planes.length + " planes");
        int channelOffset = 0;
        int outputStride = 1;

        for (int i = 0; i < planes.length; i++) {
            switch (i) {
                case 0:
                    channelOffset = 0;
                    outputStride = 1;
                    break;
                case 1:
                    if (colorFormat == COLOR_FormatI420) {
                        channelOffset = width * height;
                        outputStride = 1;
                    } else if (colorFormat == COLOR_FormatNV21) {
                        channelOffset = width * height + 1;
                        outputStride = 2;
                    }
                    break;
                case 2:
                    if (colorFormat == COLOR_FormatI420) {
                        channelOffset = (int) (width * height * 1.25);
                        outputStride = 1;
                    } else if (colorFormat == COLOR_FormatNV21) {
                        channelOffset = width * height;
                        outputStride = 2;
                    }
                    break;
            }

            ByteBuffer buffer = planes[i].getBuffer();
            int rowStride = planes[i].getRowStride();
            int pixelStride = planes[i].getPixelStride();
            logDebug("pixelStride " + pixelStride);
            logDebug("rowStride " + rowStride);
            logDebug("width " + width);
            logDebug("height " + height);
            logDebug("buffer size " + buffer.remaining());
            int shift = (i == 0) ? 0 : 1;
            int w = width >> shift;
            int h = height >> shift;
            buffer.position(rowStride * (crop.top >> shift) + pixelStride * (crop.left >> shift));
            for (int row = 0; row < h; row++) {
                int length;
                if (pixelStride == 1 && outputStride == 1) {
                    length = w;
                    buffer.get(data, channelOffset, length);
                    channelOffset += length;
                } else {
                    length = (w - 1) * pixelStride + 1;
                    buffer.get(rowData, 0, length);
                    for (int col = 0; col < w; col++) {
                        data[channelOffset] = rowData[col * pixelStride];
                        channelOffset += outputStride;
                    }
                }
                if (row < h - 1) {
                    buffer.position(buffer.position() + rowStride - length);
                }
            }
            logDebug("Finished reading data from plane " + i);
        }
        return data;
    }

    public void imageToMat(Image image) {
        ByteBuffer buffer;
        int rowStride;
        int pixelStride;
        int width = image.getWidth();
        int height = image.getHeight();
        int offset = 0;

        Image.Plane[] planes = image.getPlanes();
        byte[] data = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];

        for (int i = 0; i < planes.length; i++) {
            buffer = planes[i].getBuffer();
            rowStride = planes[i].getRowStride();
            pixelStride = planes[i].getPixelStride();
            int w = (i == 0) ? width : width / 2;
            int h = (i == 0) ? height : height / 2;
            for (int row = 0; row < h; row++) {
                int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
                if (pixelStride == bytesPerPixel) {
                    int length = w * bytesPerPixel;
                    buffer.get(data, offset, length);

                    if (h - row != 1) {
                        buffer.position(buffer.position() + rowStride - length);
                    }
                    offset += length;
                } else {
                    if (h - row == 1) {
                        buffer.get(rowData, 0, width - pixelStride + 1);
                    } else {
                        buffer.get(rowData, 0, rowStride);
                    }

                    for (int col = 0; col < w; col++) {
                        data[offset++] = rowData[col * pixelStride];
                    }
                }
            }
        }

        Mat mat = new Mat(height + height / 2, width, CvType.CV_8UC1);
        mat.put(0, 0, data);
        mMatRgb = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
        Imgproc.cvtColor(mat, mMatRgb, Imgproc.COLOR_YUV2RGB_I420);
    }

    private void matToBitmap() {
        // Bitmap for recognize
        recognizeBitmap = Bitmap.createBitmap(mMatRgb.cols(), mMatRgb.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mMatRgb, recognizeBitmap);

        Mat matResize = new Mat();
        Imgproc.resize(mMatRgb, matResize, new Size(cropSize, cropSize));
        // Bitmap for detect
        detectBitmap = Bitmap.createBitmap(matResize.cols(), matResize.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matResize, detectBitmap);

    }

    private void logDebug(String msg){
        final String TAG = "---LOG TAG---";
        if (VERBOSE)
            Log.d(TAG, msg);
    }

    private void logDebug(String tag, String msg){
        if (VERBOSE)
            Log.d(tag, msg);
    }

    private Bitmap cropPanelbyLocation(final RectF location) {
        int width = recognizeBitmap.getWidth();
        int height = recognizeBitmap.getHeight();
        Bitmap panel = Bitmap.createBitmap(recognizeBitmap,
                                         (int)(location.left*width),
                                         (int)(location.top*height),
                                         (int)(location.width()*width),
                                         (int)(location.height()*height));

        width = panel.getWidth();
        height = panel.getHeight();
        // new size
        float newWidth = (float)panelSize;
        float newHeight = (float)panelSize;
        // scale ratio
        float scaleWidth = newWidth / width;
        float scaleHeight = newHeight / height;
        //
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);

        Bitmap panelResize = Bitmap.createBitmap(panel, 0, 0, width, height, matrix, true);

        return  panelResize;
    }

    /*
    private Mat imageToMat(Image image) {
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        Mat mYuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CV_8UC1);
        mYuv.put(0, 0, nv21);
        Mat mRGB = new Mat();
        cvtColor(mYuv, mRGB, Imgproc.COLOR_YUV2RGB_I420, 3);
        return mRGB;
    }*/
}
