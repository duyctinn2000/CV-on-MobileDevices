package com.example.classifcation.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import com.example.classifcation.env.Utils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;


public class MultiLabelsClassifier implements Classifier {

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final boolean isQuantized)
            throws IOException {

        final MultiLabelsClassifier d = new MultiLabelsClassifier();


        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            d.labels.add(line);
        }
        br.close();

        if (isQuantized) {
            d.IMAGE_MEAN = 0.0f;
            d.IMAGE_STD = 1.0f;
            d.PROBABILITY_MEAN = 0.0f;
            d.PROBABILITY_STD = 255.0f;
        } else {
            d.IMAGE_MEAN = 0f;
            d.IMAGE_STD = 255f;
            d.PROBABILITY_MEAN = 0.0f;
            d.PROBABILITY_STD = 1.0f;
        }

        try {
            Interpreter.Options options = (new Interpreter.Options());
            CompatibilityList compatList = new CompatibilityList();

            if(isGPU && compatList.isDelegateSupportedOnThisDevice()){
                // if the device has a supported GPU, add the GPU delegate
                d.delegateOptions = compatList.getBestOptionsForThisDevice();
                d.gpuDelegate = new GpuDelegate(d.delegateOptions);
                options.addDelegate(d.gpuDelegate);
            } else {
                // if the GPU is not supported, run on 4 threads
                options.setNumThreads(NUM_THREADS);
            }

            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Reads type and shape of input and output tensors, respectively.

        int imageTensorIndex = 0;
        int[] imageShape = d.tfLite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        d.imageSizeY = imageShape[1];
        d.imageSizeX = imageShape[2];
        DataType imageDataType = d.tfLite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape = d.tfLite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = d.tfLite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        d.inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        d.outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
        d.probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();



        return d;
    }

    private static TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
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

    private static float IMAGE_MEAN = 0f;

    private static float IMAGE_STD = 255f;

    private static final int MAX_RESULTS = 3;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static float PROBABILITY_MEAN = 0.0f;

    private static float PROBABILITY_STD = 1.0f;

    private GpuDelegate gpuDelegate = null;
    private GpuDelegate.Options delegateOptions;


    private TensorImage inputImageBuffer;

    /** Output probability TensorBuffer. */
    private TensorBuffer outputProbabilityBuffer;

    /** Processer to apply post processing of the output probability. */
    private TensorProcessor probabilityProcessor;

    //config yolo
    private int imageSizeY;
    private int imageSizeX;

    private final String TAG = "ActionsClassifier";

    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private static boolean isGPU = false;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    public static float inferenceTime;

    // Config values.

    // Pre-allocated buffers.
    private List<String> labels = new ArrayList<>();

    private Interpreter tfLite;

    private MultiLabelsClassifier() {
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        // TODO: Define an ImageProcessor from TFLite Support Library to do preprocessing
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        Trace.beginSection("recognizeImage");

        Trace.beginSection("loadImage");
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        inputImageBuffer = loadImage(bitmap);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        Trace.endSection();
        Log.i(TAG,"Timecost to load the image: " + String.valueOf(endTimeForLoadImage - startTimeForLoadImage));

        // Runs the inference call.
        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        // TODO: Run TFLite inference
        tfLite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();

        inferenceTime = endTimeForReference - startTimeForReference + endTimeForLoadImage - startTimeForLoadImage;

        Log.i(TAG,"Timecost to run model inference: " + String.valueOf(endTimeForReference - startTimeForReference));

        // Gets the map of label and probability.
        // TODO: Use TensorLabel from TFLite Support Library to associate the probabilities
        //       with category labels

        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        Trace.endSection();

        // Gets top-k results.
        return getResults(labeledProbability);

    }

    private static List<Recognition> getResults(Map<String, Float> labelProb)
    {
        Recognition temp;
        final ArrayList<Recognition> recognitions = new ArrayList<>();
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        int index = 0;
        float total = 0.0F;
        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
            total += entry.getValue();
            if (index==25) {
                temp = pq.poll();
                temp.setConfidence(temp.getConfidence()/total);
                if (temp.getTitle().equals("etc")) {
                    recognitions.add(pq.poll());
                } else {
                    recognitions.add(temp);
                }
                for (int j = 0; j < 2; j++) {
                    temp = pq.poll();
                    temp.setConfidence(temp.getConfidence()/total);
                    recognitions.add(temp);
                }
                total = 0.0F;
                pq.clear();
            } else if (index==42) {
                for (int j = 0; j < 3; j++) {
                    temp = pq.poll();
                    temp.setConfidence(temp.getConfidence()/total);
                    recognitions.add(temp);
                }
                total = 0.0F;
                pq.clear();
            } else if (index==48) {
                for (int j = 0; j < 3; j++) {
                    temp = pq.poll();
                    temp.setConfidence(temp.getConfidence()/total);
                    recognitions.add(temp);
                }
                total = 0.0F;
                pq.clear();
            } else if (index==50) {
                for (int j = 0; j < 2; j++) {
                    temp = pq.poll();
                    temp.setConfidence(temp.getConfidence()/total);
                    recognitions.add(temp);
                }
                total = 0.0F;
                pq.clear();
            }
            index++;
        }
        return recognitions;
    }
}