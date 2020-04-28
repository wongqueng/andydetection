package com.example.andydetection;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MYFACE {


    private static final String MODEL_FILE = "file:///android_asset/mypb.pb";
    private static final String INPUT = "input";
    private static final String Placeholder_1 = "Placeholder_1";
    private static final String OUTPUT = "outputt";
    private static final int IMAGE_SIZE = 36;
    private static final String TAG = "MYFACE";
    private AssetManager assetManager;
    private TensorFlowInferenceInterface inferenceInterface;

    MYFACE(AssetManager mgr) {
        assetManager = mgr;
        loadModel();
    }

    private boolean loadModel() {
        //AssetManager
        try {
            inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
            Log.d("Facenet", "[*]load model success");
        } catch (Exception e) {
            Log.e("Facenet", "[*]load model failed" + e);
            return false;
        }
        return true;
    }

    //读取Bitmap像素值，预处理(/255)，转化为一维数组返回
    private float[] normalizeImage(Bitmap bitmap) {
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        float[] floatValues = new float[w * h * 3];
        int[] intValues = new int[w * h];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        float imageStd = 255f;

        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = ((val >> 16) & 0xFF) / imageStd;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / imageStd;
            floatValues[i * 3 + 2] = (val & 0xFF) / imageStd;
        }
        return floatValues;
    }


    public float[] classfier(Bitmap bitmap) {
        long start= System.currentTimeMillis();
        Bitmap bm = ThumbnailUtils.extractThumbnail(bitmap, IMAGE_SIZE, IMAGE_SIZE);
        float[] PNetIn = normalizeImage(bm);
        inferenceInterface.feed(INPUT, PNetIn, 1, IMAGE_SIZE, IMAGE_SIZE, 3);
        inferenceInterface.feed(Placeholder_1, new float[]{0f});
        String[] outputNames = new String[]{OUTPUT};
        inferenceInterface.run(outputNames);
        float[] prob = new float[4];
        inferenceInterface.fetch(OUTPUT, prob);
        long end= System.currentTimeMillis();
        Log.d(TAG, "[" + prob[0] + ";" + prob[1] + "];costtime="+(end-start));
        return prob;
    }


}
