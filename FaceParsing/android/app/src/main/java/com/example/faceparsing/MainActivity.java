package com.example.faceparsing;
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


import com.example.faceparsing.env.Utils;
import com.example.faceparsing.tflite.Classifier;
import com.example.faceparsing.tflite.YoloV5Classifier;

import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends Activity {

    private static final String DOG_IMAGE = "dog_face.jpg";

    private static String TF_OD_API_MODEL_FILE =  "new_old_deeplab.tflite";// "unet_2.tflite";
    private static String TF_OD_API_DEEPLAB_MODEL_FILE = "new_deeplab.tflite";

    private static final String YOLO_MODEL = "best-fp16.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labels.txt";


    private UNetSeg segmenter;

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
//        imageView_below = findViewById(R.id.imageView_below);
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
                        "com.example.faceparsing.fileprovider",
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
            segmenter = new UNetSeg(getAssets(),TF_OD_API_MODEL_FILE,TF_OD_API_DEEPLAB_MODEL_FILE,TF_OD_API_LABELS_FILE,YOLO_MODEL);
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

                    final List<Bitmap> results = segmenter.recognizeImage(rgbFrameBitmap);
//                    final List<Classifier.Recognition> face_results = detector.recognizeImage(rgbFrameBitmap);

//                    final List<FaceParsing> end2endResults = end2endDetector.recognizeImage(rgbFrameBitmap);
                    inferenceTime = SystemClock.uptimeMillis() - startTime;
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            handleResult(results);
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


    private void handleResult(List<Bitmap> results) {
        tv = new StringBuilder();

        if (results.size()>0) {
            imageView_mid.setImageBitmap(results.get(0));
        }
//        if (results.size()>1) {
//            imageView_below.setImageBitmap(results.get(1));
//        }
//        imageView.setImageBitmap(detectionBitmap);
        tv.append("Yolo inference time: "+ String.format("%.3fs", UNetSeg.yoloInferenceTime / 1000.0f));
        tv.append("\nDeeplab inference time: "+ String.format("%.3fs", UNetSeg.inferenceTime / 1000.0f));
        tv.append("\nPostprocessing time: "+ String.format("%.3fs", UNetSeg.postInferenceTime / 1000.0f));
        tv.append("\nTotal time: "+ String.format("%.3fs", (UNetSeg.postInferenceTime+UNetSeg.inferenceTime+UNetSeg.yoloInferenceTime) / 1000.0f));

        resultText.setText(tv);

//        imageView.setImageBitmap(croppedImage);
    }


    public File getPhotoFile() {
        File filesDir = getFilesDir();
        return new File(filesDir, fileName);
    }
}

