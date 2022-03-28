package com.example.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Log;

import com.example.segmentation.env.Utils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class UnetSeg {
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = false;

    private GpuDelegate gpuDelegate = null;
    private GpuDelegate.Options delegateOptions;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    private Interpreter tfLite;
    private List<String> labels = new ArrayList<>();
    public static float inferenceTime,preTime,postTime;

    private final float IMAGE_MEAN = 0f;

    private final float IMAGE_STD = 255.0f;

    private int INPUT_SIZE = 128;
    private TensorImage inputImageBuffer;
    private ByteBuffer segmentationMasks,imgData;
    private int NUM_CLASSES;
    private int [] segmentColors;
    private long [][][] outputMask;
    private int [] intValues;
    private float inp_scale;
    private int inp_zero_point;
    private float oup_scale;
    private int oup_zero_point;


    public UnetSeg(final AssetManager assetManager, final String modelFilename, final String labelFilename) throws IOException {
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }
        br.close();

        NUM_CLASSES = 3;
//        NUM_CLASSES=3;
        segmentColors = new int[NUM_CLASSES];

        Random random = new Random(System.currentTimeMillis());
        segmentColors[0] = Color.TRANSPARENT;
        segmentColors[1] = Color.RED;
        segmentColors[2] = Color.GREEN;


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

        Tensor inpten = tfLite.getInputTensor(0);
        inp_scale = inpten.quantizationParams().getScale();
        inp_zero_point = inpten.quantizationParams().getZeroPoint();
        Tensor oupten = tfLite.getOutputTensor(0);
        oup_scale = oupten.quantizationParams().getScale();
        oup_zero_point = oupten.quantizationParams().getZeroPoint();

        DataType imageDataType = tfLite.getInputTensor(0).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);


        imgData = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());
        intValues = new int[INPUT_SIZE * INPUT_SIZE];

        segmentationMasks = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 2 * 4);
        segmentationMasks.order(ByteOrder.nativeOrder());
//        outputMask = new long[1][INPUT_SIZE][INPUT_SIZE];

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


    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                    // Quantized model
//                    imgData.put((byte) ((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
//                    imgData.put((byte) ((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
//                    imgData.put((byte) (((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f);
                imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);
                imgData.putFloat((pixelValue & 0xFF) / 255.0f);

            }
        }
        return imgData;
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
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap maskBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, conf);
        Bitmap tempBitmap = Utils.scaleBitmapAndKeepRatio(bitmap, INPUT_SIZE, INPUT_SIZE);
//        ByteBuffer byteBuffer_ = convertBitmapToByteBuffer(tempBitmap);
        TensorImage tensorImage = loadImage(tempBitmap);
        int[][] mSegmentBits = new int[INPUT_SIZE][INPUT_SIZE];
        Map<Integer, Object> outputMap = new HashMap<>();

        Object[] inputArray = {tensorImage.getBuffer()};
        segmentationMasks.rewind();
        outputMap.put(0, segmentationMasks);
        long endPre = SystemClock.uptimeMillis();
        preTime = endPre - startTimeForLoadImage;
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        long endInfer = SystemClock.uptimeMillis();
        inferenceTime = endInfer - endPre;

        ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
        byteBuffer.rewind();

        int[] maskIntValues = new int[maskBitmap.getWidth()*maskBitmap.getHeight()];

        for (int y = 0; y< INPUT_SIZE;y++) {
            for (int x = 0; x< INPUT_SIZE;x++) {
                float maxVal = 0f;
                mSegmentBits[x][y] = (int) 0;

                    for (int c = 0; c< 2;c++) {
                        float value = segmentationMasks.getFloat((y * INPUT_SIZE * 2 + x * 2 + c) * 4);
//                    float value = segmentationMasks.getFloat((y * INPUT_SIZE * 3 + x * 3 + c) * 4);
//                    float value = oup_scale * (((int) (segmentationMasks.get() & 0xFF) - oup_zero_point);
//                        float value = oup_scale * (((int) byteBuffer.get() & 0xFF) - oup_zero_point);
//                        float value =  byteBuffer.get();
                        if (c == 0 || value > maxVal) {
                            maxVal = value;
                            mSegmentBits[x][y] = c;
                        }
                }

                    if(mSegmentBits[x][y]==0) {
                        maskIntValues[y*INPUT_SIZE + x] = Color.BLACK;
//                        maskBitmap.setPixel(x, y, Color.WHITE);
//                        resultBitmap.setPixel(x,y,bitmap.getPixel(x,y));
                    } else {
                        maskIntValues[y*INPUT_SIZE + x] = Color.WHITE;
//                        maskBitmap.setPixel(x, y, Color.BLACK);
                    }
//                    else if (mSegmentBits[x][y]==2) {
//                        maskBitmap.setPixel(x, y, Color.GREEN);
//                    }
//                resultBitmap.setPixel(x,y,segmentColors[mSegmentBits[x][y]]);
//                resultBitmap.setPixel(x,y,segmentColors[mSegmentBits[x][y]]);
//                System.out.println(maxVal);

//                maskBitmap.setPixel(x, y, Color.WHITE);
//                resultBitmap.setPixel(x,y,segmentColors[mSegmentBits[x][y]]);

//                if (maxVal>0.5) {
////                    System.out.println(maxVal);
////                String label = labelsArrays[mSegmentBits[x][y]];
////                    int color = segmentColors[mSegmentBits[x][y]];
////                itemsFound.put(label, color)
////                    int newPixelColor =
////                            ColorUtils.compositeColors(
////                                    segmentColors[mSegmentBits[x][y]],
////                                    scaledBitmap.getPixel(x, y)
////                            );
//
//                    maskBitmap.setPixel(x, y, Color.WHITE);
//                    resultBitmap.setPixel(x,y,bitmap.getPixel(x,y));
//                }


            }
        }
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
        postTime = endTimeForLoadImage-endInfer;

        return resultBitmap;
    }


}
