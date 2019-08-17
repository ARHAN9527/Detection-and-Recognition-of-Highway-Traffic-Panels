package com.example.panelapplication;

import android.content.Intent;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import com.example.panelapplication.env.Logger;
import com.example.panelapplication.tflite.TFLiteObjectDetectionAPIModel;
import com.example.panelapplication.tflite.TFLiteRecognitionAPIModel;

import java.io.IOException;
import java.util.Locale;

public class SurfaceViewActivity extends AppCompatActivity {
    private VideoSurfaceView mVsv;

    private static final Logger LOGGER = new Logger();
    private static final String VIDEO_FILE = "demo_阿羅哈客運 3999 線路 國道一號 中山高速公路 高雄 - 台北 全程 路程景CUT_panel.mp4";
    // Configuration values for the prepackaged SSD model.
    private static final String TF_OD_API_MODEL_FILE = "saved_model_mobilev2_ssd_width0125_38-64000.tflite"; // saved_model_mobilev2_ssd_width0125_38_extra05re
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    // Configuration values for the prepackaged Recognition model.
    private static final String TF_OR_API_MODEL_FILE = "saved_model_recognition_model_mobilev2_width0125.tflite";
    private static final String TF_OR_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final int TF_OR_API_INPUT_SIZE = 128;
    private static final boolean TF_OR_API_IS_QUANTIZED = false;

    private TextToSpeech tts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_surface_view);

        //設定隱藏標題
        getSupportActionBar().hide();
        //設定隱藏狀態
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN);

        mVsv = findViewById(R.id.vsv_1);

        createLanguageTTS();
        mVsv.tts = tts;

        Intent intent = getIntent();
        String fileName = intent.getStringExtra("fileName");
        mVsv.setVideoFile(fileName);

        try {
            mVsv.setDetector(TFLiteObjectDetectionAPIModel.create(getAssets(),
                                                                 TF_OD_API_MODEL_FILE,
                                                                 TF_OD_API_LABELS_FILE,
                                                                 TF_OD_API_INPUT_SIZE,
                                                                 TF_OD_API_IS_QUANTIZED));

            mVsv.setRecognition(TFLiteRecognitionAPIModel.create(getAssets(),
                                                                TF_OR_API_MODEL_FILE,
                                                                TF_OR_API_LABELS_FILE,
                                                                TF_OR_API_INPUT_SIZE,
                                                                TF_OR_API_IS_QUANTIZED));


        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast = Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        mVsv.setIsRunning(true);
    }


    private void createLanguageTTS()
    {
        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener(){
            @Override
            public void onInit(int status) {
                // TTS 初始化成功
                if( status == TextToSpeech.SUCCESS )
                {
                    // 指定的語系: 英文(美國)
                    Locale locale = Locale.TAIWAN;

                    if( tts.isLanguageAvailable(locale) == TextToSpeech.LANG_COUNTRY_AVAILABLE )
                    {
                        tts.setLanguage(locale);
                        tts.setPitch(1);    //語調(1為正常語調；0.5比正常語調低一倍；2比正常語調高一倍)
                        tts.setSpeechRate(2);    //速度(1為正常速度；0.5比正常速度慢一倍；2比正常速度快一倍)
                    }
                }
            }
        });
    }
}
