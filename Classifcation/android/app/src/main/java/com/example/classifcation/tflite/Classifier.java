package com.example.classifcation.tflite;

import android.graphics.Bitmap;
import android.graphics.RectF;

import java.util.List;

/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Classifier {
    List<Recognition> recognizeImage(Bitmap bitmap);

    void close();

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private Float confidence;

        /**
         * Optional location within the source image for the location of the recognized object.
         */
        private RectF location;


        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }


        public void setConfidence(Float confidence) {
            this.confidence = confidence;
        }

        public Float getConfidence() {
            return confidence;
        }

        public String getTitle() {return title;}

        @Override
        public String toString() {
            String resultString = "";

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }
}
