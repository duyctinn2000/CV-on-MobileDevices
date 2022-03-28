package com.example.classifcation;
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
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import com.example.classifcation.env.Utils;
import com.example.classifcation.tflite.CustomImageClassifier;
import com.example.classifcation.tflite.Classifier;
import com.example.classifcation.tflite.End2EndClassifier;
import com.example.classifcation.tflite.ExistsLabelsClassifier;
import com.example.classifcation.tflite.MultiLabelsClassifier;

import org.json.JSONException;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.vision.classifier.Classifications;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class MainActivity extends Activity {

    private static final String DOG_IMAGE = "dog.png";

    private static String TF_OD_API_MODEL_FILE = "dog-cat-existed-model.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/dog-cat-existed-labels.txt";
    private static boolean TF_OD_API_IS_QUANTIZED = false;

    private Classifier detector;
    private End2EndClassifier end2endDetector;

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
    private ImageView imageView;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        verifyStoragePermissions(this);
        mFile = getPhotoFile();


        setContentView(R.layout.activity_main);
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
                        "com.example.classifcation.fileprovider",
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

            detector = ExistsLabelsClassifier.create(getAssets(),TF_OD_API_MODEL_FILE,TF_OD_API_LABELS_FILE,TF_OD_API_IS_QUANTIZED);

//            end2endDetector = new End2EndClassifier(this,TF_OD_API_MODEL_FILE);

        } catch (final IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Exception initializing classifier!" + e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        this.rgbFrameBitmap = Utils.getBitmapFromAsset(MainActivity.this, DOG_IMAGE);

        imageView.setImageBitmap(this.rgbFrameBitmap);

        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Handler handler = new Handler();

                new Thread(() -> {
                    startTime = SystemClock.uptimeMillis();

                    final List<Classifier.Recognition> results = detector.recognizeImage(rgbFrameBitmap);
//                    final List<Classifications> end2endResults = end2endDetector.recognizeImage(rgbFrameBitmap);
                    inferenceTime = SystemClock.uptimeMillis() - startTime;
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                handleExistedResult(results);
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

    private void handleExistedResult (List < Classifier.Recognition > results) throws JSONException {
        boolean dogExisted = false;
        boolean catExisted = false;
        StringBuilder tv = new StringBuilder();
        for (final Classifier.Recognition result : results) {
//            tv.append(result.getTitle()+String.format(": %.3f\n", result.getConfidence()));
            if (result.getConfidence()>0.5) {
                tv.append(result.getTitle()+":           True\n");
                if (result.getTitle().equals("Dog")) {
                    dogExisted = true;
                }
                else if (result.getTitle().equals("Cat")) {
                    catExisted = true;
                }
            } else {
                tv.append(result.getTitle()+":           False\n");
            }
        }
//        if (dogExisted && catExisted) {
//            tv.append("Both Cat and Dog exist");
//        } else if (dogExisted) {
//            tv.append("Only Dog exist");
//        } else if (catExisted) {
//            tv.append("Only Cat exist");
//        } else {
//            tv.append("Not exist");
//        }
        tv.append("\n\nInference Time: "  + String.format("%.3fs", ExistsLabelsClassifier.inferenceTime / 1000.0f));
        resultText.setText(tv);

    }


    private void handleResult (List < Classifier.Recognition > results, List<Classifications> end2endResults) throws JSONException {
        String tv = "Support Model ";
        tv = tv +"- Run Time: " + String.format("%.4fs", (CustomImageClassifier.preTime+CustomImageClassifier.inferenceTime+CustomImageClassifier.postTime) / 1000.0f) + "\n\n";
        for (final Classifier.Recognition result : results) {
            Log.i(TAG,result.toString());
            tv = tv + result.toString() +"\n";
        }
        tv = tv +"\nTask Model ";
        tv = tv +"- Run Time: " + String.format("%.4fs", (End2EndClassifier.runTime) / 1000.0f) + "\n\n";
        List<Category> categories = end2endResults.get(0).getCategories();
        for (final Category category : categories) {
            Log.i(TAG,categories.toString());
            tv = tv + category.getLabel() + " " + String.format("(%.1f", (category.getScore()) * 100.0f)  +"%)\n";
        }
//        tv = tv+"\nPre Time: "  + String.format("%.4fs", CustomImageClassifier.preTime / 1000.0f);
//        tv = tv+"\nModel Time: "  + String.format("%.4fs", CustomImageClassifier.inferenceTime / 1000.0f);
//        tv = tv+"\nPost Time: "  + String.format("%.4fs", CustomImageClassifier.postTime / 1000.0f);
        resultText.setText(tv);
    }
    private void handleMultiResult (List < Classifier.Recognition > results) throws JSONException {
        StringBuilder tv = new StringBuilder();
        int index = 0;
        for (final Classifier.Recognition result : results) {
            if (index == 0) {
                tv.append("Situation: ");
            } else if (index == 3) {
                tv.append("\n\nAction: ");
            } else if (index == 6) {
                tv.append("\n\nEmotion: ");
            } else if (index == 9){
                tv.append("\n\nLocation: ");
            }
            index+=1;
            Log.i(TAG,result.toString());
            tv.append(result.toString() +", ");
        }
        tv.append("\n\nInference Time: "  + String.format("%.3fs", MultiLabelsClassifier.inferenceTime / 1000.0f));
        resultText.setText(tv);

    }

    public File getPhotoFile() {
        File filesDir = getFilesDir();
        return new File(filesDir, fileName);
    }
}

