package facedetection;

import org.opencv.core.*;
import org.opencv.core.Point;  
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class FaceDetector {
    private static JFrame frame;
    private static JLabel imageLabel;
    private static CascadeClassifier faceDetector;
    private static FaceDatabase faceDB;

    static {
        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Initialize face detector and database
        faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");
        faceDB = new FaceDatabase("resources/known_faces");
        
        createGUI();
        startCamera();
    }

    private static void createGUI() {
        frame = new JFrame("Face Recognition");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        
        imageLabel = new JLabel();
        frame.add(imageLabel, BorderLayout.CENTER);
        
        JPanel buttonPanel = new JPanel();
        JButton saveButton = new JButton("Save My Face");
        saveButton.addActionListener((ActionEvent e) -> {
            saveCurrentFace();
        });
        
        buttonPanel.add(saveButton);
        frame.add(buttonPanel, BorderLayout.SOUTH);
        frame.setVisible(true);
    }

    private static void startCamera() {
        VideoCapture camera = new VideoCapture(0);
        Mat frameMat = new Mat();
        
        while(frame.isVisible()) {
            camera.read(frameMat);
            if(!frameMat.empty()) {
                detectAndRecognizeFaces(frameMat);
                displayImage(frameMat);
            }
            try { Thread.sleep(30); } catch (InterruptedException ex) {}
        }
        camera.release();
    }

    private static void detectAndRecognizeFaces(Mat frame) {
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(frame, faceDetections);
        
        for (Rect rect : faceDetections.toArray()) {
            // Draw rectangle around face
            Imgproc.rectangle(
                frame, 
                new Point(rect.x, rect.y), 
                new Point(rect.x + rect.width, rect.y + rect.height),
                new Scalar(0, 255, 0), 
                3
            );
            
            // Recognize face
            Mat face = new Mat(frame, rect);
            String name = faceDB.recognize(face);
            
            // Display name
            Imgproc.putText(
                frame, 
                name, 
                new Point(rect.x, rect.y - 10), 
                Imgproc.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                new Scalar(0, 255, 0), 
                2
            );
        }
    }

    private static void saveCurrentFace() {
        String name = JOptionPane.showInputDialog("Enter your name:");
        if(name != null && !name.isEmpty()) {
            Mat frame = new Mat();
            VideoCapture tempCam = new VideoCapture(0);
            tempCam.read(frame);
            
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(frame, faceDetections);
            
            if(faceDetections.toArray().length > 0) {
                Rect rect = faceDetections.toArray()[0];
                Mat face = new Mat(frame, rect);
                faceDB.saveFace(face, name);
                JOptionPane.showMessageDialog(null, "Face saved successfully!");
            }
            tempCam.release();
        }
    }

    private static void displayImage(Mat mat) {
        BufferedImage image = matToBufferedImage(mat);
        imageLabel.setIcon(new ImageIcon(image));
        frame.repaint();
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if(mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        
        byte[] bytes = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, bytes);
        
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(bytes, 0, targetPixels, 0, bytes.length);
        return image;
    }
}