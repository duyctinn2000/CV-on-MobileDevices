package com.example.android.tflitecamerademo;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.FileProvider;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_IMAGE_CAPTURE = 100;
    private static final int REQUEST_IMAGE_SELECT = 200;
    private static final String IMAGE_DOG = "dog3.jpg";
    private static final String fileName = "output.jpg";
    private Button btnCamera;
    private Button btnSelect, btnRotateLeft;
    private Button btnPlay;
    private ImageView ivCaptured;
    private TextView tvLabel, timeLabel;
    private Bitmap bmp;
    private Captioner captioner;
    private Captioner lstm_captioner;
    private Attention_Captioner attention_captioner;
    private TextToSpeech tts;
    private String imgPath;
    private File mFile;


    // Storage Permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    /**
     * Checks if the app has permission to write to device storage
     *
     * If the app does not has permission then the user will be prompted to grant permissions
     *
     * @param activity
     */
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
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        verifyStoragePermissions(this);
        setContentView(R.layout.activity_main);
        tts = new TextToSpeech(this.getApplicationContext(), new TextToSpeech.OnInitListener()
        {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    tts.setLanguage(Locale.US);
                    tts.setPitch(1.3f);
                    tts.setSpeechRate(1f);
                }
            }
        });

        mFile = getPhotoFile();
        final Intent captureImage = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        bmp = getBitmapFromAsset(MainActivity.this, IMAGE_DOG);

        ivCaptured = (ImageView) findViewById(R.id.ivCaptured);
        ivCaptured.setImageBitmap(resize(bmp));
        tvLabel = (TextView) this.findViewById(R.id.tvlabel);
        timeLabel = (TextView) findViewById(R.id.infereceTime);

        btnRotateLeft = (Button) findViewById(R.id.id_btnRotateLeft);
        btnRotateLeft.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bmp != null) {
                    Matrix matrix = new Matrix();
                    matrix.postRotate(90);
                    Bitmap rotatedBitmap = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
                    bmp = rotatedBitmap;
                    ivCaptured.setImageBitmap(resize(bmp));
                }
            }
        });

        btnPlay = (Button) this.findViewById(R.id.id_btnPlay);
        btnPlay.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                String attention_result = captioner.predictImage(bmp);
                String lstm_result = captioner.predictImage(bmp);
                onTaskCompleted("lstm_result",lstm_result);
            }
        });


        btnCamera = (Button) this.findViewById(R.id.id_btnCamera);
        btnCamera.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                initPrediction();
                Uri uri = FileProvider.getUriForFile(MainActivity.this,
                        "com.example.android.tflitecamerademo.fileprovider",
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

        btnSelect = (Button) this.findViewById(R.id.id_btnSelect);
        btnSelect.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                initPrediction();
                Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, REQUEST_IMAGE_SELECT);
            }
        });

        try
        {
            captioner = new Captioner(this);
//            lstm_captioner = new Captioner(this);
//            attention_captioner = new Attention_Captioner(this);
        }
        catch (Exception e)
        {

        }

    }

    private static Bitmap resize(Bitmap image) {
        int maxHeight = 2000;
        int maxWidth = 2000;
        if (maxHeight > 0 && maxWidth > 0) {
            int width = image.getWidth();
            int height = image.getHeight();
            float ratioBitmap = (float) width / (float) height;
            float ratioMax = (float) maxWidth / (float) maxHeight;

            int finalWidth = maxWidth;
            int finalHeight = maxHeight;
            if (ratioMax > ratioBitmap) {
                finalWidth = (int) ((float)maxHeight * ratioBitmap);
            } else {
                finalHeight = (int) ((float)maxWidth / ratioBitmap);
            }
            image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true);
            return image;
        } else {
            return image;
        }
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

            bmp = BitmapFactory.decodeFile(imgPath);

            ivCaptured.setImageBitmap(resize(bmp));

        } else {
            btnCamera.setEnabled(true);
            btnSelect.setEnabled(true);
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    private void initPrediction() {
//        btnCamera.setEnabled(false);
//        btnSelect.setEnabled(false);
        tvLabel.setText("");
        timeLabel.setText("");
        tvLabel.setBackgroundColor(Color.WHITE);
    }

    private void speak_out() {
        String caption=tvLabel.getText().toString();
        tts.speak(caption, TextToSpeech.QUEUE_ADD, null,null);
    }
    /**
     * display the results on screen
     */
    public void onTaskCompleted(String attention_result, String lstm_result) {
//        String inferTime = "Attention: " + attention_result.split("\t")[1] + "\nLSTM: " + lstm_result.split("\t")[1];
//        String caption = "Attention: "+attention_result.split("\t")[0] + "\nLSTM: "+lstm_result.split("\t")[0];
        String inferTime = lstm_result.split("\t")[1];
        String caption = lstm_result.split("\t")[0];
        Log.i("q23",caption);
        tvLabel.setText(caption);
        timeLabel.setText(inferTime);

        tvLabel.setText("Caption: " + caption);
        timeLabel.setText("Inference Time: "+inferTime+"s");
        tvLabel.setBackgroundColor(Color.rgb(255, 255, 80));
        tts.speak(caption, TextToSpeech.QUEUE_ADD, null,null);
        btnCamera.setEnabled(true);
        btnSelect.setEnabled(true);
    }

    /**
     * Create a file Uri for saving an image or video
     */
    public File getPhotoFile() {
        File filesDir = getFilesDir();
        return new File(filesDir, fileName);
    }


    @Override
    public void onDestroy()
    {
        super.onDestroy();
        captioner.close();
//        lstm_captioner.close();
//        attention_captioner.close();
    }
}