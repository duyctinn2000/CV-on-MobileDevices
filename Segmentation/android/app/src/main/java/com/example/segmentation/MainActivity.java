package com.example.segmentation;
import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.database.Cursor;
import android.graphics.Bitmap;

import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.provider.MediaStore;

import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;


import com.example.segmentation.env.Utils;

import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class MainActivity extends Activity {

    private static final String DOG_IMAGE = "dog.png";

    private static String TF_OD_API_MODEL_FILE = "new_deeplabv3_256.tflite";
//    private static String TF_OD_API_MODEL_FILE = "mv2_unet.tflite";//"u2net_coco.tflite";
//    private static String TF_OD_API_MODEL_FILE = "u2netp_coco_oiv6_e136.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labels.txt";

    private DeeplabSeg deeplabSeg;
    private U2NetSeg u2NetSeg;

    private static final String fileName = "output.jpg";

    private Bitmap rgbFrameBitmap = null;

    long startTime, inferenceTime;

    private static final int REQUEST_IMAGE_SELECT = 200;
    private static final int REQUEST_IMAGE_CAPTURE = 0;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;

    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private File mFile;


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
    private ImageView imageView,imageViewBelow;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        verifyStoragePermissions(this);
        mFile = getPhotoFile();

        setContentView(R.layout.activity_main);
        imageViewBelow = findViewById(R.id.imageView_below);
        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);
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
                        "com.example.segmentation.fileprovider",
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
            deeplabSeg = new DeeplabSeg(getAssets(),TF_OD_API_MODEL_FILE,TF_OD_API_LABELS_FILE);
        } catch (IOException e) {
            e.printStackTrace();
        }

//        u2NetSeg = new U2NetSeg(getAssets(),TF_OD_API_MODEL_FILE);

        this.rgbFrameBitmap = Utils.getBitmapFromAsset(MainActivity.this, DOG_IMAGE);

        imageView.setImageBitmap(this.rgbFrameBitmap);

        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Handler handler = new Handler();

                new Thread(() -> {
                    startTime = SystemClock.uptimeMillis();

                    final Bitmap result = deeplabSeg.recognizeImage(rgbFrameBitmap);
//                    final Pair<Bitmap,Bitmap> results = u2NetSeg.recognizeImage(rgbFrameBitmap);
                    //                    final List<Classifications> end2endResults = end2endDetector.recognizeImage(rgbFrameBitmap);
                    inferenceTime = SystemClock.uptimeMillis() - startTime;
                    Log.i("InferenceTime", String.format("%.3fs", inferenceTime / 1000.0f));
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                handleResult(result);
//                                handleResult(results);
                            } catch (JSONException e) {
                                e.printStackTrace();
                            }
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

            int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

            switch(orientation) {

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

    private void handleResult (Bitmap result) throws JSONException {

        StringBuilder tv = new StringBuilder();
//        imageView.setImageBitmap(result);
        imageViewBelow.setImageBitmap(result);

//        tv.append("Pre-process Time: "  + String.format("%.5fs", UnetSeg.preTime / 1000.0f));
        tv.append("\nInference Time: "  + String.format("%.5fs", DeeplabSeg.inferenceTime / 1000.0f));
//        tv.append("\nPost-Process Time: "  + String.format("%.5fs", UnetSeg.postTime / 1000.0f));
//        tv.append("\nTotal Time: "  + String.format("%.5fs", (UnetSeg.postTime+UnetSeg.preTime+UnetSeg.inferenceTime) / 1000.0f));
        resultText.setText(tv);
    }

    private void handleResult (Pair<Bitmap,Bitmap> result) throws JSONException {

        StringBuilder tv = new StringBuilder();
        imageView.setImageBitmap(result.first);
        imageViewBelow.setImageBitmap(result.second);

        tv.append("\n\nInference Time: "  + String.format("%.5fs", U2NetSeg.inferenceTime / 1000.0f));
        resultText.setText(tv);
    }


    public File getPhotoFile() {
        File filesDir = getFilesDir();
        return new File(filesDir, fileName);
    }
}

