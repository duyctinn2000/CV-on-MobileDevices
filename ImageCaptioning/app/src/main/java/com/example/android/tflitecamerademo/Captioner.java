package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.util.Log;

import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;


import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Captioner {

    private static final String TAG = "TfLiteCameraDemo";
    private final Interpreter.Options tfliteOptions_lstm = (new Interpreter.Options());
    private final Interpreter.Options tfliteOptions = (new Interpreter.Options());
    // The loaded TensorFlow Lite model.
    private MappedByteBuffer tfliteModel;
    private MappedByteBuffer tfliteModel_lstm;
    // An instance of the driver class to run model inference with Tensorflow Lite.
    protected Interpreter tflite;
    protected Interpreter tflite_lstm;
    // A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.
    private long [] input_feed = null;
    private float [][] state_feed = null;
    private static final int NUM_THREADS = 2;

    private float [][] softmax;
    private float [][] lstm_state;
    private float [][] initial_state;

    private final float IMAGE_MEAN = 0;

    private final float IMAGE_STD = 256.0f;
    private final int IMAGE_SIZE = 346;
    private ByteBuffer imgData;
    private int[] intValues;


    Vocabulary vocabulary;

    Captioner(Activity activity) throws IOException {

        Interpreter.Options options = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();


        String inceptionModelPath = getInceptionModelPath();
        String lstmModelPath = getLSTMModelPath();
        tfliteOptions.setNumThreads(NUM_THREADS);
//        if(compatList.isDelegateSupportedOnThisDevice()){
//            Log.i("Hello","123");
//            // if the device has a supported GPU, add the GPU delegate
//            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
//            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
//            tfliteOptions_lstm.addDelegate(gpuDelegate);
//        } else {
//            // if the GPU is not supported, run on 4 threads
//            tfliteOptions_lstm.setNumThreads(NUM_THREADS);
//        }
        tfliteOptions_lstm.setNumThreads(NUM_THREADS);
        tfliteModel = loadModelFile(activity, inceptionModelPath);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        tfliteModel_lstm = loadModelFile(activity, lstmModelPath);
        tflite_lstm = new Interpreter(tfliteModel_lstm, tfliteOptions_lstm);


        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");

        imgData = ByteBuffer.allocateDirect(1 * IMAGE_SIZE * IMAGE_SIZE * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());
        intValues = new int[IMAGE_SIZE * IMAGE_SIZE];

        input_feed = new long[1];
        state_feed = new float[1][1024];

        softmax = new float[1][12000];
        lstm_state = new float[1][1024];
        initial_state = new float[1][1024];

        vocabulary = new Vocabulary(activity);

    }

    // Classifies a frame from the preview stream.
    String classifyFrame() {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            //builder.append(new SpannableString("Uninitialized Classifier."));
        }
        //convertBitmapToByteBuffer(bitmap);
        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        String caption = runInference();
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        // Print the results.
        long duration = endTime - startTime;

        return caption + "\t" + String.valueOf((float) duration/1000);
    }

    // Closes tflite to release resources.
    public void close() {
        tflite.close();
        tflite = null;
        tfliteModel = null;

        tflite_lstm.close();
        tflite_lstm = null;
        tfliteModel_lstm = null;

    }


    // Memory-map the model file in Assets.
    private MappedByteBuffer loadModelFile(Activity activity, String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String getInceptionModelPath() {
        return "inceptionv3.tflite";
    }

    private String getLSTMModelPath() {
        return "lstm.tflite";
    }


    private String runInference(){
        Map<Integer, Object> outputs_cnn = new TreeMap<Integer, Object>();
        Map<Integer, Object> outputs_lstm = new TreeMap<Integer, Object>();
        Object [] inputs_cnn = new Object[1];
        Object [] inputs_lstm = new Object[2];
        inputs_cnn[0] = imgData;
        inputs_lstm[0] = input_feed;
        inputs_lstm[1] = state_feed;

        Log.d(TAG, "inputs complete");

        outputs_lstm.put(tflite_lstm.getOutputIndex("import/softmax"), softmax);
        outputs_lstm.put(tflite_lstm.getOutputIndex("import/lstm/state"), lstm_state);
        outputs_cnn.put(tflite.getOutputIndex("import/lstm/initial_state"), initial_state);

        Log.d(TAG, "outputs complete");
        tflite.runForMultipleInputsOutputs(inputs_cnn, outputs_cnn);


        int maxCaptionLength = 20;
        List<Integer> words = new ArrayList<Integer>();
        // TODO - replace with vocab.start_id
        words.add(vocabulary.getStartIndex());

        for(int i=0;i<initial_state[0].length;i++)
        {
            state_feed[0][i] = initial_state[0][i];
        }
        input_feed[0] = words.get(words.size() - 1);

        int maxIdx;
        for(int i=0;i<maxCaptionLength;i++)
        {
            tflite_lstm.runForMultipleInputsOutputs(inputs_lstm, outputs_lstm);
            maxIdx = findMaximumIdx(softmax[0]);
            words.add(maxIdx);
            // TODO - replace with vocab.end_id
            if(maxIdx == vocabulary.getEndIndex())
            {
                break;
            }

            input_feed[0] = (long)maxIdx;
            for(int j=0;j<state_feed[0].length;j++)
            {
                state_feed[0][j] = lstm_state[0][j];
            }
            Log.d(TAG, "current word: " + maxIdx);
            //state_feed[0] = ;
        }

        Log.d(TAG, "run inference complete");
        //Log.d(TAG, "result= " + Arrays.deepToString(initial_state));
        StringBuilder sb = new StringBuilder();
        for(int i=1;i<words.size()-1;i++)
        {
            sb.append(vocabulary.getWordAtIndex(words.get(i)));
            sb.append(" ");
        }

        return sb.toString();
    }


    private int findMaximumIdx(float [] arr)
    {
        int n = arr.length;
        float maxval = arr[0];
        int maxidx = 0;
        for(int i=1;i<n;i++)
        {
            if(maxval < arr[i])
            {
                maxval = arr[i];
                maxidx = i;
            }
        }
        return maxidx;
    }

    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                int pixelValue = intValues[i * IMAGE_SIZE + j];

                imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);

            }
        }
        return imgData;
    }

    public String predictImage(Bitmap origImg) {
        //Log.d(TAG, "predictImage: imgPath=" + imgPath);
        //Bitmap origImg = BitmapFactory.decodeFile(imgPath);
        Bitmap img_346 = Utils.processBitmap(origImg, IMAGE_SIZE);
        imgData = convertBitmapToByteBuffer(img_346);

        return classifyFrame();
    }

    private static byte[] stringToBytes(String s) {
        return s.getBytes(StandardCharsets.US_ASCII);
    }
}


class Vocabulary
{

    List<String> id2word;
    Vocabulary(Activity activity)
    {
        try
        {
            id2word = loadWords(activity);
        }
        catch(Exception e)
        {

        }

    }

    public String getLabelPath()
    {
        return "word_counts.txt";
    }

    public List<String> loadWords(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
        String line;
        String [] parts;
        while ((line = reader.readLine()) != null) {
            parts = line.split(" ");
            labelList.add(parts[0]);
        }
        reader.close();
        return labelList;
    }

    public int getStartIndex()
    {
        return 1;
    }

    public int getEndIndex()
    {
        return 2;
    }

    public String getWordAtIndex(int index)
    {
        return id2word.get(index);
    }

}