package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;


import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Attention_Captioner {

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
    private float [][][] decoder_feed = null;
    private float [][][][] output_cnn = null;
    private float [][] decoder_predict = null;
    private float [][] decoder_hidden = null;
    private float [][][] decoder_attention = null;


    private final boolean isQuantized = false;

    private static final int NUM_THREADS = 4;

    private static float IMAGE_MEAN = 0f;

    private static float IMAGE_STD = 255f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static float PROBABILITY_MEAN = 0.0f;

    private static float PROBABILITY_STD = 1.0f;

    private TensorImage inputCNNBuffer;

    /** Output probability TensorBuffer. */
    private TensorBuffer outputCNNBuffer;

    private TensorBuffer inputEncoderBuffer;

    private TensorBuffer outputEncoderBuffer;

    private TensorBuffer outputDecoderBuffer;

    /** Processer to apply post processing of the output probability. */
    private TensorProcessor outputProcessor;

    //config yolo
    private int imageSizeY;
    private int imageSizeX;


    Attention_Vocal vocab;

    Attention_Captioner(Activity activity) throws IOException {

        String encoderModelPath = getEncoderModelPath();
        String decoderModelPath = getDecoderModelPath();
        String cnnModelPath = getCNNModelPath();

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

        if (isQuantized) {
            IMAGE_MEAN = 0.0f;
            IMAGE_STD = 1.0f;
            PROBABILITY_MEAN = 0.0f;
            PROBABILITY_STD = 255.0f;
        } else {
            IMAGE_MEAN = 127.5f;
            IMAGE_STD = 127.5f;
            PROBABILITY_MEAN = 0.0f;
            PROBABILITY_STD = 1.0f;
        }

        int imageTensorIndex = 0;
        int[] imageShape = tflite_cnn.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite_cnn.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] outputCNNShape = tflite_cnn.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType outputCNNType = tflite_cnn.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputCNNBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputCNNBuffer = TensorBuffer.createFixedSize(outputCNNShape, outputCNNType);

        // Creates the post processor for the output probability.
        outputProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        output_cnn = new float[1][8][8][2048];

        input_encoder = new float[1][64][2048];

        decoder_feed = new float[1][64][256];
        input_feed = new int[1][1];
        hiden_feed = new float[1][512];

        decoder_predict = new float[1][4369];//[9196];
        decoder_hidden = new float[1][512];
        decoder_attention = new float[1][64][1];

        //loadImage(activity);
        vocab = new Attention_Vocal(activity);

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


    private static TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
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

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputCNNBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        // TODO: Define an ImageProcessor from TFLite Support Library to do preprocessing
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.BILINEAR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputCNNBuffer);
    }


    private String runInference(){

        tflite_cnn.run(inputCNNBuffer.getBuffer(),output_cnn);
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
        inputCNNBuffer = loadImage(origImg);
        return classifyFrame();
    }

}


class Attention_Vocal
{

    List<String> id2word;
    Attention_Vocal(Activity activity)
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