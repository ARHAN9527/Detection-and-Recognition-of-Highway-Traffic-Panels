package com.example.panelapplication;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;

public class MainActivity extends AppCompatActivity {

    private Button mBtn;
    //private MyView mMv;
    private EditText mEt;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mBtn = findViewById(R.id.btn_1);
        //mSv = findViewById(R.id.sv_1);
        //mMv = findViewById(R.id.mv_1);
        mEt = findViewById(R.id.et_1);

        //solve "No implementation found for long org.opencv.core.Mat.n_Mat"
        Loader.load(opencv_java.class);

        mBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //String fileName = mEt.getText().toString();
                //mSv.fileName = fileName;
                //mSv.fileFormat= fileName.substring(fileName.lastIndexOf(".")+1);
                //mVsv.setVideoFile(mEt.getText().toString());
                Intent intent = new Intent(MainActivity.this, SurfaceViewActivity.class);
                intent.putExtra("fileName", mEt.getText().toString());
                startActivity(intent);
            }
        });
    }
}
