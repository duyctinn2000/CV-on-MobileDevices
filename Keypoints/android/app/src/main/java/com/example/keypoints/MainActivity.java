package com.example.keypoints;
import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.database.Cursor;
import android.graphics.Bitmap;

import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.provider.MediaStore;

import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;


import com.example.keypoints.env.Utils;
import com.example.keypoints.tflite.Classifier;
import com.example.keypoints.tflite.YoloV5Classifier;

import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends Activity {

    private static final String DOG_IMAGE = "2-dogs.jpeg";

    private static String TF_OD_API_MODEL_FILE = "keypoints_new.tflite";

    private static final String YOLO_MODEL = "yolov5n_224.tflite";

    private ImageKeypoints segmenter;

    private Classifier detector;

    private static final String fileName = "output.jpg";

    private Bitmap rgbFrameBitmap = null;

    long startTime, inferenceTime;

    private static final int REQUEST_IMAGE_SELECT = 200;
    private static final int REQUEST_IMAGE_CAPTURE = 0;

    public static float MINIMUM_CONFIDENCE_TF_OD_API = 0.7f;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;

    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private File mFile;

    StringBuilder tv;


    private String imgPath;

    public static void verifyStoragePermissions(Activity activity) {
        // Check if we have write permission
        int permission1 = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (permission1 != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    private static final String TAG = "MainActivity";

    private Button detectButton, galleryButton, cameraButton;
    private ImageView imageView, imageView_below, imageView_mid;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        verifyStoragePermissions(this);
        mFile = getPhotoFile();

        setContentView(R.layout.activity_main);
        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);
        imageView_below = findViewById(R.id.imageView_below);
        imageView_mid = findViewById(R.id.imageView_mid);
        galleryButton = findViewById(R.id.galleryButton);
        resultText = findViewById(R.id.result);
        cameraButton = findViewById(R.id.btn_camera);

        galleryButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                resultText.setText("");
                Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, REQUEST_IMAGE_SELECT);
            }
        });

        final Intent captureImage = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        cameraButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Uri uri = FileProvider.getUriForFile(MainActivity.this,
                        "com.example.keypoints.fileprovider",
                        mFile);
                captureImage.putExtra(MediaStore.EXTRA_OUTPUT, uri);
                List<ResolveInfo> cameraActivities = getPackageManager().queryIntentActivities(captureImage,
                        PackageManager.MATCH_DEFAULT_ONLY);
                for (ResolveInfo activity : cameraActivities) {
                    grantUriPermission(activity.activityInfo.packageName,uri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
                }
                startActivityForResult(captureImage, REQUEST_IMAGE_CAPTURE);
            }
        });

        try {
            detector =
                    YoloV5Classifier.create(
                            getAssets(),
                            YOLO_MODEL);
        } catch (final IOException e) {
            e.printStackTrace();
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        try {
            segmenter = new ImageKeypoints(getAssets(),TF_OD_API_MODEL_FILE);
        } catch (IOException e) {
            e.printStackTrace();
        }

        this.rgbFrameBitmap = Utils.getBitmapFromAsset(MainActivity.this, DOG_IMAGE);

        imageView.setImageBitmap(this.rgbFrameBitmap);

        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Handler handler = new Handler();

                new Thread(() -> {
                    startTime = SystemClock.uptimeMillis();

//                    final float[][] result = segmenter.recognizeImage(rgbFrameBitmap);
                    final List<Classifier.Recognition> face_results = detector.recognizeImage(rgbFrameBitmap);

//                    final List<Keypoints> end2endResults = end2endDetector.recognizeImage(rgbFrameBitmap);
                    inferenceTime = SystemClock.uptimeMillis() - startTime;
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            handleResult(face_results);
                        }
                    });
                }).start();
            }
        });
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        if ((requestCode == REQUEST_IMAGE_CAPTURE || requestCode == REQUEST_IMAGE_SELECT) && resultCode == RESULT_OK) {

            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                imgPath = mFile.getPath();
            } else {
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                Cursor cursor = MainActivity.this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgPath = cursor.getString(columnIndex);
                cursor.close();
            }

            rgbFrameBitmap = BitmapFactory.decodeFile(imgPath);

            ExifInterface ei = null;
            try {
                ei = new ExifInterface(imgPath);
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

                switch (orientation) {

                    case ExifInterface.ORIENTATION_ROTATE_90:

                        this.rgbFrameBitmap = rotateImage(rgbFrameBitmap, 90);
                        break;

                    case ExifInterface.ORIENTATION_ROTATE_180:

                        this.rgbFrameBitmap = rotateImage(rgbFrameBitmap, 180);
                        break;

                    case ExifInterface.ORIENTATION_ROTATE_270:

                        this.rgbFrameBitmap = rotateImage(rgbFrameBitmap, 270);
                        break;

                }
            } catch (RuntimeException e) {

            }
            imageView.setImageBitmap(rgbFrameBitmap);


        } else {
            cameraButton.setEnabled(true);
            galleryButton.setEnabled(true);
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    public static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    private Bitmap handleKeypointResult(List<float[][]> results, List<RectF> locations, boolean ok) {
        Bitmap tempBitmap = rgbFrameBitmap.copy(rgbFrameBitmap.getConfig(),true);
        final Canvas canvas = new Canvas(tempBitmap);
        final Paint paint = new Paint();

        for (int i = 0; i<results.size(); i++) {
            float[][] result = results.get(i);
            RectF location = locations.get(i);
            paint.setColor(Color.RED);
            canvas.drawCircle(location.left+result[0][0], location.top+result[0][1] , 10, paint);
            paint.setColor(Color.BLUE);
            canvas.drawCircle(location.left+result[1][0], location.top+result[1][1] , 10, paint);
            paint.setColor(Color.YELLOW);
            canvas.drawCircle(location.left+result[2][0], location.top+result[2][1] , 10, paint);
            paint.setColor(Color.CYAN);
            canvas.drawCircle(location.left+result[3][0], location.top+result[3][1] , 10, paint);
            paint.setColor(Color.GREEN);
            canvas.drawCircle(location.left+result[4][0], location.top+result[4][1] , 10, paint);
        }
        return tempBitmap;
    }

    private void handleResult(List<Classifier.Recognition> results) {
        tv = new StringBuilder();
        Bitmap tempBitmap = rgbFrameBitmap.copy(rgbFrameBitmap.getConfig(),true);
        Bitmap detectionBitmap = rgbFrameBitmap.copy(rgbFrameBitmap.getConfig(),true);
        final Canvas canvas = new Canvas(detectionBitmap);
        final Paint paint = new Paint();
        final Paint fgPaint;
        final float textSizePx;
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);
        fgPaint = new Paint();

        fgPaint.setStyle(Paint.Style.STROKE);
        fgPaint.setColor(Color.BLUE);
        fgPaint.setStrokeWidth(2f);
        
        long timeKeypoint = 0;

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();


        List<float[][]> keypoint_results = new ArrayList<>();
        List<RectF> keypoint_locations = new ArrayList<>();

        tv.append("Yolo: " + String.format("%.4fs",YoloV5Classifier.inferenceTime/1000));



        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();

            location.top = location.top * tempBitmap.getHeight() / YoloV5Classifier.imageHeight;
            location.bottom = location.bottom * tempBitmap.getHeight() / YoloV5Classifier.imageHeight;
            location.left = location.left * tempBitmap.getWidth() / YoloV5Classifier.imageWidth;
            location.right = location.right * tempBitmap.getWidth() / YoloV5Classifier.imageWidth;
            canvas.drawRect(location, fgPaint);


            float location_width = location.width();
            float location_height = location.height();

            if (location.width() > location.height()) {
                location.top -= (location_width-location_height)/2;
                location.bottom += (location_width-location_height)/2;
            } else if (location.height() > location.width()) {
                location.left -= (location_height-location_width)/2;
                location.right += (location_height-location_width)/2;
            }
            float zoomValue_W = 60*location.width()/Math.min(tempBitmap.getHeight(),tempBitmap.getWidth());
            float zoomValue_H = 60*location.height()/Math.min(tempBitmap.getHeight(),tempBitmap.getWidth());
            if (location.top-zoomValue_H<0 && location.top>=0) {
                zoomValue_H = location.top;
            }
            if (location.left-zoomValue_W<0 && location.left>=0) {
                zoomValue_W = location.left;
            }
            if (location.right+zoomValue_W>tempBitmap.getWidth() && tempBitmap.getWidth()-location.right>=0 && zoomValue_W>tempBitmap.getWidth()-location.right) {
                zoomValue_W = tempBitmap.getWidth()-location.right;
            }
            if (location.bottom+zoomValue_H>tempBitmap.getHeight() && tempBitmap.getHeight()-location.bottom>=0 && zoomValue_H>tempBitmap.getHeight()-location.bottom) {
                zoomValue_H = location.height()-location.bottom;
            }
            location.top -= zoomValue_H;
            location.bottom += zoomValue_H;
            location.left -= zoomValue_W;
            location.right += zoomValue_W;


            if (location.top<0){
                location.top=0;
            }
            if (location.bottom>tempBitmap.getHeight()) {
                location.bottom=tempBitmap.getHeight();
            }
            if (location.left<0){
                location.left=0;
            }
            if (location.right>tempBitmap.getWidth()) {
                location.right=tempBitmap.getWidth();
            }


            canvas.drawRect(location, paint);

            Bitmap croppedImage = Bitmap.createBitmap(tempBitmap, (int) location.left, (int) location.top, (int) location.width(), (int) location.height());

            float[][] keypoint_result = segmenter.recognizeImage(croppedImage);
//            Log.i("123213",""+ImageKeypoints.inferenceTime);
            timeKeypoint += ImageKeypoints.inferenceTime;

            for (int i = 0; i < keypoint_result.length; i++) {
                keypoint_result[i][0] = keypoint_result[i][0] * croppedImage.getWidth() / 224;
                keypoint_result[i][1] = keypoint_result[i][1] * croppedImage.getHeight() / 224;
            }

            keypoint_results.add(keypoint_result);

            keypoint_locations.add(location);

        }

        tv.append("\nKeypoint: " + String.format("%.4fs", timeKeypoint / 1000.0f));

        List<float[][]> full_keypoint_results = new ArrayList<>();
        List<RectF> full_keypoint_locations = new ArrayList<>();

        float[][] keypoint_result = segmenter.recognizeImage(tempBitmap);

        for (int i = 0; i < keypoint_result.length; i++) {
            keypoint_result[i][0] = keypoint_result[i][0] * tempBitmap.getWidth() / 224;
            keypoint_result[i][1] = keypoint_result[i][1] * tempBitmap.getHeight() / 224;
        }

        full_keypoint_results.add(keypoint_result);

        full_keypoint_locations.add(new RectF(0,0,0,0));

        imageView_mid.setImageBitmap(handleKeypointResult(keypoint_results,keypoint_locations,true));
//        imageView_below.setImageBitmap(handleKeypointResult(full_keypoint_results,full_keypoint_locations,false));
        imageView.setImageBitmap(detectionBitmap);

        resultText.setText(tv);

    }


    public File getPhotoFile() {
        File filesDir = getFilesDir();
        return new File(filesDir, fileName);
    }
}

