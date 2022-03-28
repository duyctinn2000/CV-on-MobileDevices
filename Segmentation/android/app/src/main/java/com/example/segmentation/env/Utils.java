package com.example.segmentation.env;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.Log;
import android.util.Pair;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    /**
     * Memory-map the model file in Assets.
     */
    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
//            return bitmap.copy(Bitmap.Config.ARGB_8888,true);
        } catch (IOException e) {
            // handle exception
            Log.e("getBitmapFromAsset", "getBitmapFromAsset: " + e.getMessage());
        }

        return bitmap;
    }

    public static Bitmap scaleBitmapAndKeepRatio(Bitmap targetBmp, int reqHeightInPixels, int reqWidthInPixels) {
        if (targetBmp.getHeight() == reqHeightInPixels && targetBmp.getWidth() == reqWidthInPixels) {
            return targetBmp;
        } else {
            Matrix matrix = new Matrix();
            matrix.setRectToRect(new RectF(0.0F, 0.0F, (float)targetBmp.getWidth(), (float)targetBmp.getHeight()), new RectF(0.0F, 0.0F, (float)reqWidthInPixels, (float)reqHeightInPixels), Matrix.ScaleToFit.FILL);
            Bitmap scaledBitmap = Bitmap.createBitmap(targetBmp, 0, 0, targetBmp.getWidth(), targetBmp.getHeight(), matrix, true);
            return scaledBitmap;
        }
    }

    public static Pair<Bitmap,Bitmap> convertArrayToBitmap(float[][][][] imageArray, int imageWidth, int imageHeight) {

        // Convert multidimensional array to 1D
        float maxValue = 0;
        float minValue = 10000f;

        for (int m=0; m < imageArray[0].length; m++) {
            for (int x=0; x < imageArray[0][0].length; x++) {
                for (int y=0; y < imageArray[0][0][0].length; y++) {
                    if (maxValue<imageArray[0][m][x][y]) {
                        maxValue = imageArray[0][m][x][y];
                    }
                    if (minValue>imageArray[0][m][x][y]) {
                        minValue = imageArray[0][m][x][y];
                    }
                }
            }
        }
        Log.i("min",""+ minValue);
        Log.i("max",""+ maxValue);


        Bitmap blackWhiteImage = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        Bitmap gradientImage = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);

        // Use manipulation like Colab post processing......  // 255 * (depth - depth_min) / (depth_max - depth_min)
        for (int x = 0; x < imageArray[0][0].length; x++) {
            for (int y = 0;  y < imageArray[0][0][0].length; y++) {

                // Create black and transparent bitmap based on pixel value above a certain number eg. 150
                // make all pixels black in case value of grayscale image is above 150
                float pixel = 255 * (imageArray[0][0][x][y] - minValue) / (maxValue - minValue);
                if (pixel>70) {
                    blackWhiteImage.setPixel(y,x,Color.WHITE);
                } else {
                    blackWhiteImage.setPixel(y,x,Color.BLACK);
                }

                gradientImage.setPixel(y,x,Color.rgb((int)pixel,(int)pixel,(int)pixel));

            }
        }
        return new Pair<>(blackWhiteImage,gradientImage);
    }


    public static float[][][][] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] intValues = new int[width*height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
        float[][][][] fourDimensionalArray = new float[1][3][width][height];

        for (int i=0; i < width ; ++i) {
            for (int j=0; j < height ; ++j) {
                int pixelValue = intValues[i*width+j];
                fourDimensionalArray[0][0][i][j] = (float)
                        Color.red(pixelValue);
                fourDimensionalArray[0][1][i][j] = (float)
                        Color.green(pixelValue);
                fourDimensionalArray[0][2][i][j] = (float)
                        Color.blue(pixelValue);
            }
        }


        // Convert multidimensional array to 1D
        float maxValue = 0;
        for (int m=0; m < fourDimensionalArray[0].length; m++) {
            for (int x=0; x < fourDimensionalArray[0][0].length; x++) {
                for (int y=0; y < fourDimensionalArray[0][0][0].length; y++) {
                    if (maxValue<fourDimensionalArray[0][m][x][y]) {
                        maxValue = fourDimensionalArray[0][m][x][y];
                    }
                }
            }
        }

        Log.i("max",""+ maxValue);

        float [][][][] finalFourDimensionalArray = new float[1][3][width][height];

        for (int i = 0; i<width; ++i) {
            for (int j = 0; j<height; ++j) {
                int pixelValue = intValues[i * width + j];
                finalFourDimensionalArray[0][0][i][j] =
                        ((Color.red(pixelValue) / maxValue) - 0.485f) / 0.229f;
                finalFourDimensionalArray[0][1][i][j] =
                        ((Color.green(pixelValue) / maxValue) - 0.456f) / 0.224f;
                finalFourDimensionalArray[0][2][i][j] =
                        ((Color.blue(pixelValue) / maxValue) - 0.406f) / 0.225f;
            }

        }

        return finalFourDimensionalArray;
    }


}
