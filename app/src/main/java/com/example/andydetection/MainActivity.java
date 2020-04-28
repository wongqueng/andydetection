package com.example.andydetection;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.opencv.imgproc.Imgproc.rectangle;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    TextView tv_result;
    JavaCameraView jcv_opencv;
    CascadeClassifier cascadeClassifier;
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);

    Mat mRgba;
    Bitmap roibm;
    MYFACE myface;
    float[] result;
    Handler mainHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            switch (msg.what) {
                case 0:
                    tv_result.setText("");
                    break;
                case 1:
                    tv_result.setText("刘德华：" + result[0] + "\n"
                            + "吴彦祖：" + result[1]+"\n"
                            + "新垣结衣：" + result[2]+"\n"
                            + "石原里美：" + result[3]+"\n"
                    );
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        tv_result = findViewById(R.id.tv_result);
        myface = new MYFACE(getAssets());
        OpenCVLoader.initDebug();
        new MyThread().start();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                    || checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
            } else {
                initCamera();
            }
        } else {
            initCamera();
        }
    }

    public void initCamera() {
        jcv_opencv = findViewById(R.id.jcv_opencv);
        jcv_opencv.setCvCameraViewListener(this);
        jcv_opencv.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);

        jcv_opencv.enableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (jcv_opencv != null)
            jcv_opencv.enableView();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (jcv_opencv != null)
            jcv_opencv.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (jcv_opencv != null)
            jcv_opencv.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(width, height, CvType.CV_8UC4);
        File mCascadeFile = null;
        try {

            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            if (!mCascadeFile.exists()) {
                InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                FileOutputStream os = new FileOutputStream(mCascadeFile);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        ;
        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        MatOfRect facedetect = new MatOfRect();
        cascadeClassifier.detectMultiScale(mRgba, facedetect);
        Rect[] facesArray = facedetect.toArray();
        Rect tmp = null;
        for (int i = 0; i < facesArray.length; i++) {
            if (tmp == null || tmp.area() < facesArray[i].area()) {
                tmp = facesArray[i];
            }
        }
        if (tmp != null) {
            rectangle(mRgba, tmp.tl(), tmp.br(), FACE_RECT_COLOR, 3);
            Mat roi_img = new Mat(mRgba, tmp);
            roibm = Bitmap.createBitmap(tmp.width, tmp.height, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(roi_img, roibm);
            if (handler != null) {
                mainHandler.removeMessages(0);
                handler.removeMessages(100);
                handler.sendEmptyMessage(100);
            }
        } else {
            mainHandler.sendEmptyMessageDelayed(0,500);
        }
        return mRgba;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == 1) {
            for (int grantResult : grantResults) {
                if (grantResult != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "对不起，没有权限，无法正常使用相机", Toast.LENGTH_SHORT).show();
                    return;
                }
            }
            initCamera();
        }
    }

    Handler handler;

    public class MyThread extends Thread {
        @Override
        public void run() {
            super.run();
            Looper.prepare();
            handler = new Handler() {
                @Override
                public void handleMessage(Message msg) {
                    super.handleMessage(msg);
                    result = myface.classfier(roibm);
                    mainHandler.sendEmptyMessage(1);

                }
            };
            Looper.loop();
        }
    }


}
