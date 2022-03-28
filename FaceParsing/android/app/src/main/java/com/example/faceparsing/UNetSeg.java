package com.example.faceparsing;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import androidx.core.graphics.ColorUtils;

import com.example.faceparsing.env.Utils;
import com.example.faceparsing.tflite.Classifier;
import com.example.faceparsing.tflite.YoloV5Classifier;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
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


public class UNetSeg {
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = false;

    private GpuDelegate gpuDelegate = null;
    private GpuDelegate.Options delegateOptions;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    private Interpreter tfLite;
    private MappedByteBuffer tfliteModel_deeplab;
    private Interpreter tfLite_deeplab;
    private List<String> labels = new ArrayList<>();
    public static float inferenceTime;
    public static float yoloInferenceTime;
    public static float postInferenceTime;


    private static float IMAGE_MEAN = 0f;

    private static float IMAGE_STD = 255f;
    private int imageWidth;
    private int imageHeight;
    private TensorImage inputImageBuffer;
    private ByteBuffer segmentationMasks;
    private int NUM_CLASSES;
    private int [] segmentColors;
    private int [] maskColors;
    private float [][][] outputMask;
    private static final int  SIZE = 256;
    private static final int OUTPUT_SIZE = 256;
    private Classifier detector;
    private static Map<String,Integer> itemsFound;


    public UNetSeg(final AssetManager assetManager, final String modelFilename,final String modelDeeplabFilename, final String labelFilename, final String yoloModelFileName) throws IOException {
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
        maskColors = new int[NUM_CLASSES];


        Random random = new Random(System.currentTimeMillis());
        maskColors[0] = Color.WHITE;
        maskColors[1] = Color.YELLOW;
        maskColors[2] = Color.GREEN;
        maskColors[3] = Color.RED;
        maskColors[4] = Color.BLACK;
        maskColors[5] = Color.BLUE;
        maskColors[6] = Color.MAGENTA;
        segmentColors[1] =Color.argb(
                150,
                255,
                0,
                255
        );
        segmentColors[2] = Color.argb(
                200,
                255,
                255,
                255 //0
        );
        segmentColors[3] = Color.argb(
                150,
                150,
                150,
                250
        );
        segmentColors[4] = Color.argb(
                150,
                36,
                180,
                36
        );
        segmentColors[5] = Color.argb(
                150,
                131,
                238,
                255
        );
        segmentColors[6] = Color.argb(
                150,
                255,
                226,
                62
        );
//        for (int i = 1; i < NUM_CLASSES; i++) {
//            segmentColors[i] =
//                    Color.argb(
//                            (128),
//                            getRandomRGBInt(random),
//                            getRandomRGBInt(random),
//                            getRandomRGBInt(random)
//                    );
//        }

        try {
            detector =
                    YoloV5Classifier.create(
                            assetManager,
                            yoloModelFileName);
        } catch (final IOException e) {
            e.printStackTrace();
            Log.i("ImageKeypoints","YoloV5Classifier could not be initialized");
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
            tfliteModel_deeplab = Utils.loadModelFile(assetManager,modelDeeplabFilename);
            tfLite_deeplab = new Interpreter(tfliteModel_deeplab, options);
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
//        outputMask = new float[1][OUTPUT_SIZE][OUTPUT_SIZE];
        itemsFound=new HashMap<>();

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

    private Bitmap faceParsing(Bitmap bitmap) {
        bitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
        inputImageBuffer = loadImage(bitmap);

        tfLite.run(inputImageBuffer.getBuffer(), segmentationMasks);
        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap resultBitmap = Bitmap.createBitmap(OUTPUT_SIZE, OUTPUT_SIZE, conf);

        int[][] mSegmentBits = new int[imageWidth][imageHeight];

        segmentationMasks.rewind();

        for (int y = 0; y< SIZE;y++) {
            for (int x = 0; x< SIZE;x++) {
                mSegmentBits[x][y] = 0;
                float maxVal = 0f;
                for (int c = 0; c < NUM_CLASSES;c++) {
                    float value = segmentationMasks.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4);
                    if (c == 0 || value > maxVal) {
                        maxVal = value;
                        mSegmentBits[x][y] = c;
                    }
                }
                String label = labels.get(mSegmentBits[x][y]);
                int color = segmentColors[mSegmentBits[x][y]];
                itemsFound.put(label, color);
                    int newPixelColor =
                        ColorUtils.compositeColors(
                                segmentColors[mSegmentBits[x][y]],
                                bitmap.getPixel(x, y)
                        );

//                    maskBitmap.setPixel(x, y, Color.WHITE);
                resultBitmap.setPixel(x,y,newPixelColor);


            }
        }



        return resultBitmap;
    }

    private Bitmap deepLabParsing(Bitmap original,Bitmap bitmap,int top,int left) {

//        bitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
        Bitmap tempBitmap = Bitmap.createScaledBitmap(bitmap,imageWidth,imageHeight,true);
        inputImageBuffer = loadImage(tempBitmap);

        long startTime = SystemClock.uptimeMillis();
        tfLite_deeplab.run(inputImageBuffer.getBuffer(), segmentationMasks);
        inferenceTime = SystemClock.uptimeMillis()-startTime;

        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap resultBitmap = original.copy(original.getConfig(),true);

        int[][] mSegmentBits = new int[imageWidth][imageHeight];

        segmentationMasks.rewind();

        Bitmap maskBitmap = Bitmap.createBitmap(SIZE,SIZE,conf);


        for (int y = 0; y< SIZE;y++) {
            for (int x = 0; x< SIZE;x++) {
                mSegmentBits[x][y] = 0;
                float maxVal = 0f;
                for (int c = 0; c < NUM_CLASSES;c++) {
                    float value = segmentationMasks.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4);
                    if (c == 0 || value > maxVal) {
                        maxVal = value;
                        mSegmentBits[x][y] = c;
                    }
                }
                maskBitmap.setPixel(x,y,ColorUtils.compositeColors(
                            segmentColors[mSegmentBits[x][y]],
                        tempBitmap.getPixel(x, y)));
//                maskBitmap.setPixel(x,y,maskColors[mSegmentBits[x][y]]);
//                String label = labels.get(mSegmentBits[x][y]);
//                int color = segmentColors[mSegmentBits[x][y]];
//                itemsFound.put(label, color);
//                int newPixelColor =
//                        ColorUtils.compositeColors(
//                                segmentColors[mSegmentBits[x][y]],
//                                bitmap.getPixel(x, y)
//                        );
//
////                    maskBitmap.setPixel(x, y, Color.WHITE);
//                resultBitmap.setPixel(x,y,newPixelColor);


            }
        }

//        Bitmap newBitmap = Bitmap.createScaledBitmap(maskBitmap,bitmap.getWidth(),bitmap.getHeight(),false);
//        int[][] maskArray = arrayFromBitmap(maskBitmap);
//        Log.i("123",""+bitmap.getWidth()+","+bitmap.getHeight());
//        Log.i("123",""+original.getWidth()+","+original.getHeight());
//        Log.i("123",""+maskBitmap.getWidth()+","+maskBitmap.getHeight());

        Bitmap resizedMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.getWidth(), bitmap.getHeight(), false);
        int[] tempIntValues = new int[bitmap.getWidth()*bitmap.getHeight()];
        int[] bitmapIntValue = new int[bitmap.getWidth()*bitmap.getHeight()];
//        int[] resultIntValue = new int[bitmap.getWidth()*bitmap.getHeight()];
        bitmap.getPixels(bitmapIntValue, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        resizedMask.getPixels(tempIntValues, 0, resizedMask.getWidth(), 0, 0, resizedMask.getWidth(), resizedMask.getHeight());
//        Bitmap resultBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), conf);

//        for (int i=0; i<tempIntValues.length;i++) {
//            if (tempIntValues[i]==Color.WHITE) {
//                resultIntValue[i]=bitmapIntValue[i];
//            }
////            else if (tempIntValues[i]==Color.GREEN) {
////                resultIntValue[i]=Color.rgb((int)(Color.red(bitmapIntValue[i])*scale),(int)(Color.green(bitmapIntValue[i])*scale),(int)(Color.blue(bitmapIntValue[i])*scale));
////            }
//        }

        resultBitmap.setPixels(tempIntValues, 0, bitmap.getWidth(), left, top, bitmap.getWidth(), bitmap.getHeight());

//        for (int y = 0; y< bitmap.getHeight();y++) {
//            for (int x = 0; x< bitmap.getWidth();x++) {
//                int c=0;
//                if (newBitmap.getPixel(x,y) == Color.BLACK) {
//                    c = 4;
//                } else if (newBitmap.getPixel(x,y) == Color.YELLOW) {
//                    c = 1;
//                } else if (newBitmap.getPixel(x,y) == Color.GREEN) {
//                    c = 2;
//                } else if (newBitmap.getPixel(x,y) == Color.RED) {
//                    c = 3;
//                } else if (newBitmap.getPixel(x,y) == Color.BLUE) {
//                    c = 5;
//                } else if (newBitmap.getPixel(x,y) == Color.MAGENTA) {
//                    c = 6;
//                }
//                int newPixelColor =
//                    ColorUtils.compositeColors(
//                            segmentColors[c],
//                            original.getPixel(x+left, y+top)
//                    );
//
//                resultBitmap.setPixel(x+left,y+top,newPixelColor);
//            }
//        }



        return resultBitmap;
    }

    private Bitmap newDeepLabParsing(Bitmap original,Bitmap bitmap,int top,int left) {

//        bitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
        Bitmap tempBitmap = Bitmap.createScaledBitmap(bitmap,imageWidth,imageHeight,true);
        inputImageBuffer = loadImage(tempBitmap);

        long startTime = SystemClock.uptimeMillis();
        tfLite.run(inputImageBuffer.getBuffer(), segmentationMasks);
        inferenceTime = SystemClock.uptimeMillis()-startTime;

        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap resultBitmap = original.copy(original.getConfig(),true);

        int[][] mSegmentBits = new int[imageWidth][imageHeight];

        segmentationMasks.rewind();

        Bitmap maskBitmap = Bitmap.createBitmap(SIZE,SIZE,conf);


        for (int y = 0; y< SIZE;y++) {
            for (int x = 0; x< SIZE;x++) {
                mSegmentBits[x][y] = 0;
                float maxVal = 0f;
                for (int c = 0; c < NUM_CLASSES;c++) {
                    float value = segmentationMasks.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4);
                    if (c == 0 || value > maxVal) {
                        maxVal = value;
                        mSegmentBits[x][y] = c;
                    }
                }
                maskBitmap.setPixel(x,y,ColorUtils.compositeColors(
                        segmentColors[mSegmentBits[x][y]],
                        tempBitmap.getPixel(x, y)));


            }
        }


        Bitmap resizedMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.getWidth(), bitmap.getHeight(), false);
        int[] tempIntValues = new int[bitmap.getWidth()*bitmap.getHeight()];
        int[] bitmapIntValue = new int[bitmap.getWidth()*bitmap.getHeight()];
        bitmap.getPixels(bitmapIntValue, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        resizedMask.getPixels(tempIntValues, 0, resizedMask.getWidth(), 0, 0, resizedMask.getWidth(), resizedMask.getHeight());


        resultBitmap.setPixels(tempIntValues, 0, bitmap.getWidth(), left, top, bitmap.getWidth(), bitmap.getHeight());



        return resultBitmap;
    }

    public List<Bitmap> recognizeImage(Bitmap bitmap) {
        long startTimeForLoadImage = SystemClock.uptimeMillis();

//        float aspect_ratio = (float) bitmap.getHeight()/bitmap.getWidth();
//        bitmap = Utils.scaleBitmapAndKeepRatio(bitmap, imageHeight, imageWidth);
//        inputImageBuffer = loadImage(bitmap);


        List<Bitmap> results = new ArrayList<>();

        final List<Classifier.Recognition> face_results = detector.recognizeImage(bitmap);
        long endYolo = SystemClock.uptimeMillis();
        yoloInferenceTime = endYolo - startTimeForLoadImage;
        Bitmap  parsingResult,croppedImage;


        for (final Classifier.Recognition result : face_results) {
            final RectF location = result.getLocation();

            location.top = location.top * bitmap.getHeight() / YoloV5Classifier.imageHeight;
            location.bottom = location.bottom * bitmap.getHeight() / YoloV5Classifier.imageHeight;
            location.left = location.left * bitmap.getWidth() / YoloV5Classifier.imageWidth;
            location.right = location.right * bitmap.getWidth() / YoloV5Classifier.imageWidth;

            float location_width = location.width();
            float location_height = location.height();

            if (location.width() > location.height()) {
                location.top -= (location_width-location_height)/2;
                location.bottom += (location_width-location_height)/2;
            } else if (location.height() > location.width()) {
                location.left -= (location_height-location_width)/2;
                location.right += (location_height-location_width)/2;
            }

            float zoomValue_W = 50*location.width()/Math.min(bitmap.getHeight(),bitmap.getWidth());
            float zoomValue_H = 50*location.height()/Math.min(bitmap.getHeight(),bitmap.getWidth());
//            if (location.top-zoomValue_H<0 && location.top>=0) {
//                zoomValue_H = location.top;
//            }
//            if (location.left-zoomValue_W<0 && location.left>=0) {
//                zoomValue_W = location.left;
//            }
//            if (location.right+zoomValue_W>bitmap.getWidth() && bitmap.getWidth()-location.right>=0 && zoomValue_W>bitmap.getWidth()-location.right) {
//                zoomValue_W = bitmap.getWidth()-location.right;
//            }
//            if (location.bottom+zoomValue_H>bitmap.getHeight() && bitmap.getHeight()-location.bottom>=0 && zoomValue_H>bitmap.getHeight()-location.bottom) {
//                zoomValue_H = location.height()-location.bottom;
//            }
            location.top -= zoomValue_H;
            location.bottom += zoomValue_H;
            location.left -= zoomValue_W;
            location.right += zoomValue_W;


            if (location.top<0){
                location.top=0;
            }
            if (location.bottom>bitmap.getHeight()) {
                location.bottom=bitmap.getHeight();
            }
            if (location.left<0){
                location.left=0;
            }
            if (location.right>bitmap.getWidth()) {
                location.right=bitmap.getWidth();
            }


            croppedImage = Bitmap.createBitmap(bitmap, (int) location.left, (int) location.top, (int) location.width(), (int) location.height());


//            Bitmap parsingResult = newDeepLabParsing(croppedImage);
//            parsingResult = newDeepLabParsing(bitmap,croppedImage,(int)location.top,(int)location.left);
//
//
//            results.add(parsingResult);

            parsingResult = deepLabParsing(bitmap,croppedImage,(int)location.top,(int)location.left);

            results.add(parsingResult);

        }


        long endTimeForLoadImage = SystemClock.uptimeMillis();
        postInferenceTime = endTimeForLoadImage-endYolo-inferenceTime;
        return results;
//        if (aspect_ratio>0) {
//            return Utils.scaleBitmapAndKeepRatio(resultBitmap,imageHeight, (int) (imageWidth/aspect_ratio));
//        }
//        return Utils.scaleBitmapAndKeepRatio(resultBitmap,(int) (imageHeight*aspect_ratio), imageWidth);
        //        Bitmap resizedMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.getWidth(), bitmap.getHeight(), false);
//        Log.i("12312321",""+resizedMask.getWidth()+" "+resizedMask.getHeight());
//        for (int x=0; x<resizedMask.getWidth();x++) {
//            for (int y=0;y<resizedMask.getHeight();y++){
//                if (Color.red(resizedMask.getPixel(x,y))>200) {
//                    resultBitmap.setPixel(x, y, bitmap.getPixel(x, y));
//                }
//            }
//        }

    }


}
