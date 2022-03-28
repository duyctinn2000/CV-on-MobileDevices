package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;


import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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

public class AT_Captioner {

    private static final String TAG = "TfLiteCameraDemo";
    private final Interpreter.Options tfliteOptions_encoder = (new Interpreter.Options());
    private final Interpreter.Options tfliteOptions_decoder = (new Interpreter.Options());
    private final Interpreter.Options tfliteOptions_cnn = (new Interpreter.Options());

    // The loaded TensorFlow Lite model.
    private MappedByteBuffer tfliteModel_encoder;
    private MappedByteBuffer tfliteModel_decoder;
    private MappedByteBuffer tfliteModel_cnn;

    // An instance of the driver class to run model inference with Tensorflow Lite.
    protected Interpreter tflite_encoder;
    protected Interpreter tflite_decoder, tflite_cnn;
    // A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.
    private int [][] input_feed = null;
    private float [][] hiden_feed = null;
    private float [][][] input_encoder = null;
    private float [][][][] image_feed = null;
    private float [][][] decoder_feed = null;
    private float [][][][] output_cnn = null;
    private float [][] decoder_predict = null;
    private float [][] decoder_hidden = null;
    private float [][][] decoder_attention = null;
    // Float model
    private final float IMAGE_MEAN = 127.5f; //0;

    private final float IMAGE_STD = 127.5f; //255.0f;
    private final int IMAGE_SIZE = 299;
    private ByteBuffer imgData;
    private int[] intValues;
    private Activity ac;

    private static final int NUM_THREADS = 2;


    Vocab vocab;

    AT_Captioner(Activity activity) throws IOException {

        String encoderModelPath = getEncoderModelPath();
        String decoderModelPath = getDecoderModelPath();
        String cnnModelPath = getCNNModelPath();
        ac = activity;

        tfliteOptions_cnn.setNumThreads(NUM_THREADS);
        tfliteModel_cnn = loadModelFile(activity, cnnModelPath);
        tflite_cnn = new Interpreter(tfliteModel_cnn, tfliteOptions_cnn);

        tfliteOptions_encoder.setNumThreads(NUM_THREADS);
        tfliteModel_encoder = loadModelFile(activity, encoderModelPath);
        tflite_encoder = new Interpreter(tfliteModel_encoder, tfliteOptions_encoder);

        tfliteOptions_decoder.setNumThreads(NUM_THREADS);
        tfliteModel_decoder = loadModelFile(activity, decoderModelPath);
        tflite_decoder = new Interpreter(tfliteModel_decoder, tfliteOptions_decoder);

        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");

        output_cnn = new float[1][8][8][2048];

        input_encoder = new float[1][64][2048];

        decoder_feed = new float[1][64][256];
        input_feed = new int[1][1];
        hiden_feed = new float[1][512];

        decoder_predict = new float[1][4369];//[9196];
        decoder_hidden = new float[1][512];
        decoder_attention = new float[1][64][1];
        imgData = ByteBuffer.allocateDirect(1 * IMAGE_SIZE * IMAGE_SIZE * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());
        intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        image_feed = new float[1][299][299][3];
        //loadImage(activity);
        vocab = new Vocab(activity);

    }

    // Classifies a frame from the preview stream.
    String classifyFrame() {
        if (tflite_cnn == null) {
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
        tflite_decoder.close();
        tflite_decoder = null;
        tfliteModel_decoder = null;

        tflite_encoder.close();
        tflite_encoder = null;
        tfliteModel_encoder = null;

        tflite_cnn.close();
        tflite_cnn = null;
        tfliteModel_cnn = null;

    }


    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) throws IOException {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        String output = "";
        imgData.rewind();
        Log.i("123123",ac.getExternalFilesDir("MyFileDir").toString());
        File myExternalFile = new File(ac.getExternalFilesDir("MyFileDir"),"image.txt");
        FileOutputStream fos = null;
        fos = new FileOutputStream(myExternalFile);
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                int pixelValue = intValues[i * IMAGE_SIZE + j];
                float x = (((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                float y = (((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                float z = ((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                output =  String.valueOf(x) + " " + String.valueOf(y) + " " + String.valueOf(z) + "\n";
                fos.write(output.getBytes());
                //Log.i("12321",String.valueOf(x) + " " + String.valueOf(y) + " " + String.valueOf(z));
            }
            Log.i("1231231",String.valueOf(i));
        }
        fos.close();
        //Log.i("1231",output);
        return imgData;
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

    private String getCNNModelPath() {
        return "cnn_model_v4.tflite";
    }

    private String getEncoderModelPath() {
        return "encoder_model_v4.tflite";
    }

    private String getDecoderModelPath() {
        return "decoder_model_v4.tflite";
    }

    private float[][][] reshape(float [][][][] output_cnn){
        float[][][] input_encoder = new float[1][64][2048];
        for (int x = 0; x<1; x++) {
            for (int y = 0; y<8; y++) {
                for (int z = 0; z<8; z++) {
                    for (int t = 0; t<2048; t++) {
                        input_encoder[x][(y + 1) * (z + 1) - 1][t] = output_cnn[x][y][z][t];
                    }
                }
            }
        }
        return input_encoder;
    }

    public void loadImage(Activity activity) throws IOException {


        BufferedReader reader = null;

        try {
            // use buffered reader to read line by line
            reader = new BufferedReader(new InputStreamReader(activity.getAssets().open("test.txt")));

            String line = null, output="";
            String[] numbers = null;
            // read line by line till end of file
            while ((line = reader.readLine()) != null) {
                // split each line based on regular expression having
                // "any digit followed by one or more spaces".

                numbers = line.split(" ");

                imgData.putFloat(Float.valueOf(numbers[0].trim()));
                imgData.putFloat(Float.valueOf(numbers[1].trim()));
                imgData.putFloat(Float.valueOf(numbers[2].trim()));
                float x = Float.valueOf(numbers[0].trim());
                float y = Float.valueOf(numbers[1].trim());
                float z = Float.valueOf(numbers[2].trim());
                output =  String.valueOf(x) + " " + String.valueOf(y) + " " + String.valueOf(z) + "\n";
                Log.i("123123",output);
            }
        } catch (IOException e) {
            System.err.println("Exception:" + e.toString());
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    System.err.println("Exception:" + e.toString());
                }
            }
        }


    }


    private String runInference(){


        tflite_cnn.run(imgData,output_cnn);
        input_encoder = this.reshape(output_cnn);
        tflite_encoder.run(input_encoder,decoder_feed);

        int maxCaptionLength = 20;
        List<Integer> words = new ArrayList<Integer>();

        words.add(vocab.getStartIndex());

        for(int i=0;i<hiden_feed[0].length;i++)
        {
            hiden_feed[0][i] = 0;
        }

        input_feed[0][0] = words.get(words.size() - 1);

        Object [] inputs_decoder = new Object[3];
        inputs_decoder[0] = hiden_feed;
        inputs_decoder[1] = input_feed;
        inputs_decoder[2] = decoder_feed;

        Map<Integer, Object> outputs_decoder = new TreeMap<Integer, Object>();
        outputs_decoder.put(0,decoder_predict);
        outputs_decoder.put(1,decoder_attention);
        outputs_decoder.put(2,decoder_hidden);

        int maxIdx;

        for(int i=0;i<maxCaptionLength;i++)
        {
            tflite_decoder.runForMultipleInputsOutputs(inputs_decoder, outputs_decoder);
            maxIdx = findMaximumIdx(decoder_predict[0]);
            words.add(maxIdx);
            // TODO - replace with vocab.end_id
            if(maxIdx == vocab.getEndIndex())
            {
                break;
            }

            input_feed[0][0] = (int)maxIdx;
            for(int j=0;j<hiden_feed[0].length;j++)
            {
                hiden_feed[0][j] = decoder_hidden[0][j];
            }
            Log.d(TAG, "current word: " + maxIdx);

        }

        Log.d(TAG, "run inference complete");
        //Log.d(TAG, "result= " + Arrays.deepToString(initial_state));
        StringBuilder sb = new StringBuilder();
        for(int i=1;i<words.size()-1;i++)
        {
            sb.append(vocab.getWordAtIndex(words.get(i)));
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

    public String predictImage(Bitmap origImg) {
        Bitmap resized_img = Utils.processBitmap(origImg, IMAGE_SIZE);
        try {
            imgData = convertBitmapToByteBuffer(resized_img);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classifyFrame();
    }

    private static byte[] stringToBytes(String s) {
        return s.getBytes(StandardCharsets.US_ASCII);
    }
}


class Vocab
{

    List<String> id2word;
    Vocab(Activity activity)
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
        return "words_v4.txt";
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
        return 3;
    }

    public int getEndIndex() { return 4; }

    public String getWordAtIndex(int index)
    {
        return id2word.get(index);
    }

}