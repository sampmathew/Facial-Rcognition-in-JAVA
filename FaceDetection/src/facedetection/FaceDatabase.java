package facedetection;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.File;
import java.util.*;

public class FaceDatabase {
    private final String facesFolder;
    private final HashMap<String, Mat> knownFaces = new HashMap<>();

    public FaceDatabase(String folderPath) {
        this.facesFolder = folderPath;
        new File(folderPath).mkdirs();
        loadKnownFaces();
    }

    public void saveFace(Mat face, String name) {
        String filename = facesFolder + File.separator + name + "_" + System.currentTimeMillis() + ".png";
        Imgcodecs.imwrite(filename, face);
        knownFaces.put(name, face);
    }

    public String recognize(Mat face) {
        double minDist = Double.MAX_VALUE;
        String recognized = "Unknown";
        
        for (Map.Entry<String, Mat> entry : knownFaces.entrySet()) {
            double dist = compareFaces(face, entry.getValue());
            if (dist < minDist && dist < 0.6) {
                minDist = dist;
                recognized = entry.getKey();
            }
        }
        return recognized;
    }

    private double compareFaces(Mat face1, Mat face2) {
        // Convert to grayscale if needed
        Mat gray1 = new Mat();
        Mat gray2 = new Mat();
        
        if (face1.channels() > 1) {
            Imgproc.cvtColor(face1, gray1, Imgproc.COLOR_BGR2GRAY);
        } else {
            face1.copyTo(gray1);
        }
        
        if (face2.channels() > 1) {
            Imgproc.cvtColor(face2, gray2, Imgproc.COLOR_BGR2GRAY);
        } else {
            face2.copyTo(gray2);
        }

        // Resize to same dimensions
        Size targetSize = new Size(100, 100);
        Imgproc.resize(gray1, gray1, targetSize);
        Imgproc.resize(gray2, gray2, targetSize);

        // Calculate histograms
        Mat hist1 = new Mat();
        Mat hist2 = new Mat();
        
        List<Mat> images1 = Collections.singletonList(gray1);
        List<Mat> images2 = Collections.singletonList(gray2);
        
        MatOfInt channels = new MatOfInt(0);
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        Mat mask = new Mat();

        Imgproc.calcHist(images1, channels, mask, hist1, histSize, ranges);
        Imgproc.calcHist(images2, channels, mask, hist2, histSize, ranges);

        // Normalize histograms
        Core.normalize(hist1, hist1);
        Core.normalize(hist2, hist2);

        // Compare histograms
        return Imgproc.compareHist(hist1, hist2, Imgproc.HISTCMP_CORREL);
    }

    private void loadKnownFaces() {
        File folder = new File(facesFolder);
        File[] files = folder.listFiles();
        
        if (files != null) {
            for (File file : files) {
                if (file.getName().endsWith(".png")) {
                    String name = file.getName().split("_")[0];
                    knownFaces.put(name, Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE));
                }
            }
        }
    }
}