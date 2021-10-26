package org.tensorflow.lite.examples.detection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.github.dhaval2404.imagepicker.ImagePicker;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.EfficientNetClassifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    public static float MINIMUM_CONFIDENCE_TF_OD_API = 0.4f;
    long startTime, inferenceTime;
    int numberOfObject;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraButton = findViewById(R.id.cameraButton);
        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);
        galleryButton = findViewById(R.id.galleryButton);
        resultText = findViewById(R.id.result);
        rotateButton = findViewById(R.id.btn_rotate_right);
        zoomoutButton = findViewById(R.id.btn_zoom_out);
        defaultButton = findViewById(R.id.btn_default);
        yolov4TinyButton = findViewById(R.id.yolov4_tiny);
        yolov5Button = findViewById(R.id.yolov5);
        efficientButton = findViewById(R.id.efficient_net);

        yolov4TinyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                yolov4TinyButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_primary));
                yolov5Button.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                efficientButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                MINIMUM_CONFIDENCE_TF_OD_API = 0.4f;
                TF_OD_API_INPUT_SIZE = 640; //320
                TF_OD_API_IS_QUANTIZED = false; //false;
                TF_OD_API_MODEL_FILE =  "yolov4-tiny-640-fp16.tflite";//"efficientdet_lite.tflite"; //"yolo-v5.tflite"; //" "yolov4-416-fp32.tflite";
                cropBitmap = Utils.processBitmap(defaultBitmap, TF_OD_API_INPUT_SIZE);
                defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
                imageView.setImageBitmap(cropBitmap);
                initBox();
                try {
                    detector =
                            YoloV4Classifier.create(
                                    getAssets(),
                                    TF_OD_API_MODEL_FILE,
                                    TF_OD_API_LABELS_FILE,
                                    TF_OD_API_IS_QUANTIZED);
                } catch (final IOException e) {
                    e.printStackTrace();
                    LOGGER.e(e, "Exception initializing classifier!");
                    Toast toast =
                            Toast.makeText(
                                    getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                    toast.show();
                    finish();
                }
            }
        });

        yolov5Button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                yolov4TinyButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                yolov5Button.setBackgroundColor(getResources().getColor(R.color.tfe_color_primary));
                efficientButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
                TF_OD_API_INPUT_SIZE = 320;
                TF_OD_API_IS_QUANTIZED = false; //false;
                TF_OD_API_MODEL_FILE =  "yolo-v5.tflite"; //" "yolov4-416-fp32.tflite";
                cropBitmap = Utils.processBitmap(defaultBitmap, TF_OD_API_INPUT_SIZE);
                defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
                imageView.setImageBitmap(cropBitmap);
                try {
                    detector =
                            YoloV5Classifier.create(
                                    getAssets(),
                                    TF_OD_API_MODEL_FILE,
                                    TF_OD_API_LABELS_FILE,
                                    TF_OD_API_IS_QUANTIZED);
                } catch (final IOException e) {
                    e.printStackTrace();
                    LOGGER.e(e, "Exception initializing classifier!");
                    Toast toast =
                            Toast.makeText(
                                    getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                    toast.show();
                    finish();
                }
            }
        });

        efficientButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                yolov4TinyButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                yolov5Button.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                efficientButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_primary));
                MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
                TF_OD_API_INPUT_SIZE = 640;
                TF_OD_API_IS_QUANTIZED = true; //false;
                TF_OD_API_MODEL_FILE =  "efficientdet_lite.tflite"; //" "yolov4-416-fp32.tflite";
                cropBitmap = Utils.processBitmap(defaultBitmap, TF_OD_API_INPUT_SIZE);
                defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
                imageView.setImageBitmap(cropBitmap);
                try {
                    detector =
                            EfficientNetClassifier.create(
                                    getAssets(),
                                    TF_OD_API_MODEL_FILE,
                                    TF_OD_API_LABELS_FILE,
                                    TF_OD_API_IS_QUANTIZED);
                } catch (final IOException e) {
                    e.printStackTrace();
                    LOGGER.e(e, "Exception initializing classifier!");
                    Toast toast =
                            Toast.makeText(
                                    getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                    toast.show();
                    finish();
                }
            }
        });

        defaultButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cropBitmap = defaultBitmap.copy(defaultBitmap.getConfig(), true);
                imageView.setImageBitmap(cropBitmap);
                initBox();
            }
        });

        zoomoutButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap smallBitmap = Utils.processBitmap(cropBitmap, TF_OD_API_INPUT_SIZE*90/100);
                cropBitmap = padBitmap(smallBitmap,TF_OD_API_INPUT_SIZE-smallBitmap.getWidth(),TF_OD_API_INPUT_SIZE - smallBitmap.getHeight());
                imageView.setImageBitmap(cropBitmap);
                initBox();
            }
        });

        rotateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (cropBitmap != null) {
                    Matrix matrix = new Matrix();
                    matrix.postRotate(90);
                    Bitmap rotatedBitmap = Bitmap.createBitmap(cropBitmap, 0, 0, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, matrix, true);
                    cropBitmap = rotatedBitmap;
                    imageView.setImageBitmap(cropBitmap);
                    initBox();
                }
            }
        });

        galleryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                resultText.setText("");
                ImagePicker.with(MainActivity.this)
                        .crop()	    			//Crop image(Optional), Check Customization for more option
//                        .compress(1024)			//Final image size will be less than 1 MB(Optional)
//                        .maxResultSize(1080, 1080)	//Final image resolution will be less than 1080 x 1080(Optional)
                        .start();
            }
        });


        cameraButton.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivity.class)));

        detectButton.setOnClickListener(v -> {
            Handler handler = new Handler();

            new Thread(() -> {
                startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = detector.recognizeImage(cropBitmap);
                inferenceTime = SystemClock.uptimeMillis() - startTime;
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        handleResult(cropBitmap, results);
                    }
                });
            }).start();

        });
        this.sourceBitmap = Utils.getBitmapFromAsset(MainActivity.this, "dog.png");

        this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
        this.defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
        this.imageView.setImageBitmap(cropBitmap);

        initBox();
    }

    private Bitmap padBitmap(Bitmap Src, int padding_x, int padding_y) {
        Bitmap outputimage = Bitmap.createBitmap(Src.getWidth()+padding_x/2,Src.getHeight()+padding_y/2, Bitmap.Config.ARGB_8888);
        Canvas can1 = new Canvas(outputimage);
        can1.drawColor(Color.GRAY);
        can1.drawBitmap(Src, padding_x/2, padding_y/2, null);
        Bitmap output = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE,TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas can2 = new Canvas(output);
        can2.drawColor(Color.GRAY);
        can2.drawBitmap(outputimage, 0, 0, null);
        return output;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK) {
            //Image Uri will not be null for RESULT_OK
            Uri uri = data.getData();

            try {
                this.sourceBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
            this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
            this.defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
            this.imageView.setImageBitmap(cropBitmap);
            initBox();
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Task Cancelled", Toast.LENGTH_SHORT).show();
        }

    }

    private static final Logger LOGGER = new Logger();

    public static int TF_OD_API_INPUT_SIZE = 640; //320

    private static boolean TF_OD_API_IS_QUANTIZED = false; //false;

    private static String TF_OD_API_MODEL_FILE =  "yolov4-tiny-640-fp16.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";

    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private Integer sensorOrientation = 90;

    private Classifier detector;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;

    protected int previewWidth = 0;
    protected int previewHeight = 0;

    private Bitmap sourceBitmap;
    private Bitmap cropBitmap, defaultBitmap;

    private Button cameraButton, detectButton, galleryButton, defaultButton, yolov4TinyButton, yolov5Button, efficientButton;
    private ImageView imageView;
    private TextView resultText;
    private ImageButton rotateButton, zoomoutButton;

    private void initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> tracker.draw(canvas));

        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);
    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        final Paint fgPaint;
        final float textSizePx;
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);
        numberOfObject = 0;
        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, 8, getResources().getDisplayMetrics());
        fgPaint = new Paint();
        fgPaint.setTextSize(textSizePx);
        fgPaint.setColor(Color.RED);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();
        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                numberOfObject++;
                canvas.drawText(String.format("%.2f", result.getConfidence()), (int) location.left+5, location.top+20, fgPaint);
                canvas.drawRect(location, paint);
//                cropToFrameTransform.mapRect(location);
//
//                result.setLocation(location);
//                mappedRecognitions.add(result);
            }
        }
        resultText.setText("Objects: " + String.valueOf(numberOfObject) + "\nInference time: "+String.valueOf(inferenceTime)+"ms");
//        tracker.trackResults(mappedRecognitions, new Random().nextInt());
//        trackingOverlay.postInvalidate();
        imageView.setImageBitmap(bitmap);
    }
}
