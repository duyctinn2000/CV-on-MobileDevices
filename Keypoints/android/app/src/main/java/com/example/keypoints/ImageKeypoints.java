package com.example.keypoints;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Log;

import androidx.core.graphics.ColorUtils;

import com.example.keypoints.env.Utils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class ImageKeypoints {
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = false;

    private GpuDelegate gpuDelegate = null;
    private GpuDelegate.Options delegateOptions;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    private Interpreter tfLite;
    private List<String> labels = new ArrayList<>();
    public static float inferenceTime;

    private static float IMAGE_MEAN = 0f;

    private static float IMAGE_STD = 1f;
    private static int imageWidth;
    private static int imageHeight;
    private TensorImage inputImageBuffer;
    private float [][][][] outputKeypoints;
    private static final int NUM_KEYPOINTS = 5;
    float[][] results;



    public ImageKeypoints(final AssetManager assetManager, final String modelFilename) throws IOException  {

        try {
            Interpreter.Options options = (new Interpreter.Options());
            CompatibilityList compatList = new CompatibilityList();

            if (isGPU && compatList.isDelegateSupportedOnThisDevice()) {
                // if the device has a supported GPU, add the GPU delegate
                delegateOptions = compatList.getBestOptionsForThisDevice();
                gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
            } else {
                // if the GPU is not supported, run on 4 threads
                options.setNumThreads(NUM_THREADS);
            }

            tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            tfLite = new Interpreter(tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Reads type and shape of input and output tensors, respectively.

        int imageTensorIndex = 0;
        int[] imageShape = tfLite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageHeight = imageShape[1];
        imageWidth = imageShape[2];
        DataType imageDataType = tfLite.getInputTensor(imageTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        outputKeypoints = new float[1][1][1][2*NUM_KEYPOINTS];

        // Creates the output tensor and its processor.
    }

    public void close() {
        if (tfLite != null) {
            // TODO: Close the interpreter
            tfLite.close();
            tfLite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        tfliteModel = null;
    }

    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        // TODO: Define an ImageProcessor from TFLite Support Library to do preprocessing
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(imageWidth, imageHeight, ResizeOp.ResizeMethod.BILINEAR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    public float[][] recognizeImage(Bitmap bitmap) {
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        results = new float[NUM_KEYPOINTS][2];

        inputImageBuffer = loadImage(bitmap);
        tfLite.run(inputImageBuffer.getBuffer(), outputKeypoints);

        int i = 0;
        while (i<NUM_KEYPOINTS*2) {
            for (int j = 0; j<2; j++) {
                results[(int)Math.floor(i/2)][j] = outputKeypoints[0][0][0][i]*imageWidth;
                i++;
            }
        }
        if (results[0][0]>results[1][0]) {
            swap(0,1);
        }
        if (results[2][0]>results[3][0]){
            swap(2,3);
        }
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        inferenceTime = endTimeForLoadImage-startTimeForLoadImage;
        return results;

    }

    private void swap(int x,int y){
        float temp = results[x][0];
        results[x][0]=results[y][0];
        results[y][0]=temp;
        temp = results[x][1];
        results[x][1]=results[y][1];
        results[y][1]=temp;
    }
}
