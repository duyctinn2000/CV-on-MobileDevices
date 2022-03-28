package com.example.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import androidx.core.graphics.ColorUtils;

import com.example.segmentation.env.Utils;

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


public class DeeplabSeg {
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = false;

    private GpuDelegate gpuDelegate = null;
    private GpuDelegate.Options delegateOptions;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    private Interpreter tfLite;
    private List<String> labels = new ArrayList<>();
    public static float inferenceTime;

    private static float IMAGE_MEAN = 127.5f;

    private static float IMAGE_STD = 127.5f;
    private int imageWidth;
    private int imageHeight;
    private TensorImage inputImageBuffer;
    private ByteBuffer segmentationMasks;
    private int NUM_CLASSES;
    private int [] segmentColors;
    private long [][][] outputMask;
    private static final int  SIZE = 256;
    private static final int OUTPUT_SIZE = 256;


    public DeeplabSeg(final AssetManager assetManager, final String modelFilename, final String labelFilename) throws IOException {
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }
        br.close();

        NUM_CLASSES = labels.size();
        segmentColors = new int[NUM_CLASSES];

        Random random = new Random(System.currentTimeMillis());
        segmentColors[0] = Color.TRANSPARENT;
        for (int i = 1; i < NUM_CLASSES; i++) {
            segmentColors[i] =
                    Color.argb(
                            (128),
                            getRandomRGBInt(random),
                            getRandomRGBInt(random),
                            getRandomRGBInt(random)
                    );
        }


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

        segmentationMasks = ByteBuffer.allocateDirect(1 * OUTPUT_SIZE * OUTPUT_SIZE * NUM_CLASSES * 4);
        segmentationMasks.order(ByteOrder.nativeOrder());
//        outputMask = new long[1][OUTPUT_SIZE][OUTPUT_SIZE];

        // Creates the output tensor and its processor.

    }
    private int getRandomRGBInt (Random random){
        return (int) (255 * random.nextFloat());
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
//                        .add(new ResizeOp(imageWidth, imageHeight, ResizeOp.ResizeMethod.BILINEAR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    public Bitmap recognizeImage(Bitmap bitmap) {
//        long startTimeForLoadImage = SystemClock.uptimeMillis();
        Log.i("12312321",""+bitmap.getWidth()+" "+bitmap.getHeight());

//        if (bitmap.getWidth()>300 || bitmap.getHeight()>300) {
//            bitmap = scaleBitmapAndKeepRatio(bitmap, 300, 300);
//        }
        float aspect_ratio = (float) bitmap.getHeight()/bitmap.getWidth();
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        Bitmap tempBitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
        inputImageBuffer = loadImage(tempBitmap);
        tfLite.run(inputImageBuffer.getBuffer(), segmentationMasks);
        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap maskBitmap = Bitmap.createBitmap(SIZE, SIZE, conf);
//        Bitmap resultBitmap = Bitmap.createBitmap(OUTPUT_SIZE, OUTPUT_SIZE, conf);
        int[] maskIntValues = new int[maskBitmap.getWidth()*maskBitmap.getHeight()];

        int[][] mSegmentBits = new int[imageWidth][imageHeight];

        segmentationMasks.rewind();

        for (int y = 0; y< SIZE;y++) {
            for (int x = 0; x< SIZE;x++) {
                float maxVal = 0f;
//                mSegmentBits[x][y] = (int) outputMask[0][y][x];

//                maskBitmap.setPixel(x, y, Color.BLACK);

                for (int c= 0; c< NUM_CLASSES;c++) {
                    float value = segmentationMasks.getFloat((y * SIZE * NUM_CLASSES + x * NUM_CLASSES + c) * 4);

                    if (c == 0 || value > maxVal) {
                        maxVal = value;
                        mSegmentBits[x][y] = c;
                    }
                }
//                if (labels.get(mSegmentBits[x][y]).equals("dog") || labels.get(mSegmentBits[x][y]).equals("cat") || labels.get(mSegmentBits[x][y]).equals("horse") ||  labels.get(mSegmentBits[x][y]).equals("sheep") ||  labels.get(mSegmentBits[x][y]).equals("cow")) {
//                String label = labelsArrays[mSegmentBits[x][y]];
//                    int color = segmentColors[mSegmentBits[x][y]];
//                itemsFound.put(label, color)
//                    int newPixelColor =
//                            ColorUtils.compositeColors(
//                                    segmentColors[mSegmentBits[x][y]],
//                                    scaledBitmap.getPixel(x, y)
//                            );
//                Log.i("213123",""+mSegmentBits[x][y]);
                if (mSegmentBits[x][y]==0) {
//                    Log.i("12313123",""+mSegmentBits[x][y]);
                    maskIntValues[y*SIZE + x] = Color.BLACK;
//                    maskBitmap.setPixel(x, y, Color.BLACK);
//                    resultBitmap.setPixel(x,y,bitmap.getPixel(x,y));
                }
                else {
                    maskIntValues[y*SIZE + x] = Color.WHITE;
                }
            }
        }

//        return maskBitmap;

        maskBitmap.setPixels(maskIntValues, 0, maskBitmap.getWidth(), 0, 0, maskBitmap.getWidth(), maskBitmap.getHeight());
        Bitmap resizedMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.getWidth(), bitmap.getHeight(), false);
        int[] tempIntValues = new int[bitmap.getWidth()*bitmap.getHeight()];
        int[] bitmapIntValue = new int[bitmap.getWidth()*bitmap.getHeight()];
        int[] resultIntValue = new int[bitmap.getWidth()*bitmap.getHeight()];
        bitmap.getPixels(bitmapIntValue, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        resizedMask.getPixels(tempIntValues, 0, resizedMask.getWidth(), 0, 0, resizedMask.getWidth(), resizedMask.getHeight());
        Bitmap resultBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), conf);

        for (int i=0; i<tempIntValues.length;i++) {
            if (tempIntValues[i]==Color.WHITE) {
                resultIntValue[i]=bitmapIntValue[i];
            }
        }

        resultBitmap.setPixels(resultIntValue, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());


        long endTimeForLoadImage = SystemClock.uptimeMillis();
        inferenceTime = endTimeForLoadImage-startTimeForLoadImage;
        return resultBitmap;
    }


}
