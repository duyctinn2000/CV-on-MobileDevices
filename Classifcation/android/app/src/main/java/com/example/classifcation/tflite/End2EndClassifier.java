package com.example.classifcation.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

import java.io.IOException;
import java.util.List;

public class End2EndClassifier {
    private ImageClassifier imageClassifier;
    public static long runTime;
    public End2EndClassifier(Context context, String modelFilename) {
        ImageClassifier.ImageClassifierOptions options = ImageClassifier.ImageClassifierOptions.builder()
                .setMaxResults(3)
                .setNumThreads(4)
                .build();
        try {
            imageClassifier = ImageClassifier.createFromFileAndOptions(context, modelFilename, options);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public List<Classifications> recognizeImage(Bitmap bitmap) {
        // Initialization

        long startTimeForRunModel = SystemClock.uptimeMillis();
        // Run inference
        TensorImage inputImage = TensorImage.fromBitmap(bitmap);

        List<Classifications> results = imageClassifier.classify(inputImage);
        long endTimeForRunModel = SystemClock.uptimeMillis();
        runTime = endTimeForRunModel - startTimeForRunModel;
        return results;
    }
}
