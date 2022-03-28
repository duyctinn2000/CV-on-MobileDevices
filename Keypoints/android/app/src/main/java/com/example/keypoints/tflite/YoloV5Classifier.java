package com.example.keypoints.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;


import com.example.keypoints.MainActivity;
import com.example.keypoints.env.Utils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
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
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;


public class YoloV5Classifier implements Classifier {

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename)
            throws IOException {

        final YoloV5Classifier d = new YoloV5Classifier();

        d.labels.add("face");

        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            CompatibilityList compatList = new CompatibilityList();

            if(isGPU && compatList.isDelegateSupportedOnThisDevice()){
                // if the device has a supported GPU, add the GPU delegate
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
            } else {
                // if the GPU is not supported, run on 4 threads
                options.setNumThreads(4);
            }
            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        int imageTensorIndex = 0;
        int[] imageShape = d.tfLite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        DataType imageDataType = d.tfLite.getInputTensor(imageTensorIndex).dataType();

        imageHeight = imageShape[1];
        imageWidth = imageShape[2];

        // Creates the input tensor.
        d.inputImageBuffer = new TensorImage(imageDataType);
        //yolov5s6
//        d.output_box = 6375;
        //yolov5s,yolov5n
//        d.output_box = 6300;
        //volov5s_224
        d.output_box = 3087;
//        if (modelFilename.equals("yolov5-s6.tflite")) {
//        } else {
//            d.output_box = 6300;
//        }


        int[] shape = d.tfLite.getOutputTensor(0).shape();
        int numClass = shape[shape.length - 1] - 5;
        d.numClass = numClass;
        d.outData = ByteBuffer.allocateDirect(d.output_box * (numClass + 5) * 4);
        d.outData.order(ByteOrder.nativeOrder());
        return d;
    }

    @Override
    public void close() {
        tfLite.close();
        tfLite = null;
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        tfliteModel = null;
    }

    private void recreateInterpreter() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = new GpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }
    }

    @Override
    public float getObjThresh() {
        return MainActivity.MINIMUM_CONFIDENCE_TF_OD_API;
    }

    //config yolo
    protected float mNmsThresh = 0.45f;

    private  int output_box;
    public static long no_nms;

    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = false;

    private TensorImage inputImageBuffer;

    private static final float IMAGE_MEAN = 0f;

    private static final float IMAGE_STD = 255.0f;
    public static int imageWidth;
    public static int imageHeight;

    public static float inferenceTime;



    /** holds a gpu delegate */
    GpuDelegate gpuDelegate = null;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    // Config values.

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();

    private ByteBuffer outData;

    private Interpreter tfLite;
    private int numClass;

    private YoloV5Classifier() {
    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }


    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
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

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
        long startTimeForLoadImage = SystemClock.uptimeMillis();

        inputImageBuffer = loadImage(bitmap);

        Map<Integer, Object> outputMap = new HashMap<>();

        outData.rewind();
        outputMap.put(0, outData);

        Object[] inputArray = {inputImageBuffer.getBuffer()};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
        byteBuffer.rewind();

        ArrayList<Recognition> detections = new ArrayList<Recognition>();

        float[][][] out = new float[1][output_box][numClass + 5];
        for (int i = 0; i < output_box; ++i) {
            for (int j = 0; j < numClass + 5; ++j) {
                out[0][i][j] = byteBuffer.getFloat();
            }
            // Denormalize xywh
            for (int j = 0; j < 4; ++j) {
                out[0][i][j] *= imageWidth;
            }
        }
        for (int i = 0; i < output_box; ++i){
            final int offset = 0;
            final float confidence = out[0][i][4];
            int detectedClass = -1;
            float maxClass = 0;


            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); ++c) {
                classes[c] = out[0][i][5 + c];
            }

            for (int c = 0; c < labels.size(); ++c) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass * confidence;

            if (confidenceInClass > getObjThresh()) {

                final float xPos = out[0][i][0];
                final float yPos = out[0][i][1];

                final float w = out[0][i][2];
                final float h = out[0][i][3];

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
            }
        }

        Log.d("YoloV5Classifier", "Detect End");
        final ArrayList<Recognition> recognitions = nms(detections);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        inferenceTime = endTimeForLoadImage-startTimeForLoadImage;
        return recognitions;
    }

    public ArrayList<Recognition> recognizeImageEfficientDet(Bitmap bitmap) {
        inputImageBuffer = loadImage(bitmap);
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][25][4]);
        outputMap.put(1, new float[1][25]);
        outputMap.put(2, new float[1][25]);
        outputMap.put(3, new float[1]);
        Object[] inputArray = {inputImageBuffer};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        //int gridWidth = OUTPUT_WIDTH_FULL[0];
        float[][][] bboxes = (float[][][]) outputMap.get(0);
        float[][] classes = (float[][]) outputMap.get(1);
        float[][] scores = (float[][]) outputMap.get(2);
        float[] num_detections = (float[]) outputMap.get(3);
        float[] coor;
        for (int i = 0; i < (int) num_detections[0]; i++) {
            final float[] all_label = new float[labels.size()];
            //Log.i("456", String.valueOf(scores[0][i]));
            if (scores[0][i] > getObjThresh() && classes[0][i] == 17) {
                coor = bboxes[0][i];
                final RectF rectF = new RectF(
                        Math.max(0, coor[1] * bitmap.getWidth()),
                        Math.max(0, coor[0] * bitmap.getHeight()),
                        Math.min(bitmap.getWidth() - 1, coor[3] * bitmap.getWidth()),
                        Math.min(bitmap.getHeight() - 1, coor[2] * bitmap.getHeight()));
                detections.add(new Recognition("" + i, labels.get((int) classes[0][i]), scores[0][i], rectF, (int) classes[0][i]));

            }

        }
        return detections;
    }
}