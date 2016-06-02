package image;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.net.URL;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

@SuppressWarnings("unused")
public class Align {

	static CascadeClassifier face_cascade;
	static CascadeClassifier eyes_cascade;

	/**
	 * Sample main class.
	 * TODO: turn this into a junit test
	 * @throws IOException
	 */
//	public static void main(String[] args) throws IOException {
//		//file named here must be in same location as this class file
//		BufferedImage image = ImageIO.read(Align.class.getResource("print.jpg"));
//
//		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//		Mat test = bufferedImageToMat(image);
//
//		Rect[] detected = buildFaceArray(test);
//
//		// If there are any faces found, draw a rectangle around it
//		for (int i = 0; i < detected.length; i++)
//			Core.rectangle(test, detected[i].tl(), detected[i].br(), new Scalar(0, 255, 0, 255), 3);
//
//		writeMatToJpgFile(test, "results.jpg");
//
//		processFaces(test, detected);
//	}

	public static void processFaces(Mat image, Rect[] faces) throws IOException {
		for (int i = 0; i < faces.length; i++) {
			Mat tempFace = image.submat((int) Math.round(faces[i].tl().y), (int) Math.round(faces[i].br().y),
					(int) Math.round(faces[i].tl().x), (int) Math.round(faces[i].br().x));
			Rect[] tempEyes = buildEyeArray(tempFace);
			if (tempEyes.length >= 2) {
				Point eyeA = new Point((tempEyes[0].tl().x + (tempEyes[0].width / 2)),
						(tempEyes[0].tl().y + (tempEyes[0].height / 2))); // find center of eye at tempEyes[0]
				Point eyeB = new Point((tempEyes[1].tl().x + (tempEyes[1].width / 2)),
						(tempEyes[1].tl().y + (tempEyes[1].height / 2))); // find center of eye at tempEyes[1]
				tempFace = alignEyes(tempFace, eyeA, eyeB);

				Rect nose = detectNose(tempFace);
				if (nose.tl().x != -1) {
					writeMatToJpgFile(tempFace, "face_" + i + ".jpg");
				}
			}
		}
	}

	/*
	 * Tested.
	 */
	public static Rect detectNose(Mat face) {

		// Create a grayscale image
		Mat grayscaleImage;
		grayscaleImage = new Mat(face.height(), face.width(), CvType.CV_8UC4);

		Imgproc.cvtColor(face, grayscaleImage, Imgproc.COLOR_RGBA2RGB);

		MatOfRect potentialNoses = new MatOfRect();

		CascadeClassifier nose_cascade = new CascadeClassifier();
		nose_cascade.load("Cascades/haarcascade_mcs_nose.xml");

		// Use the classifier to detect faces
		if (nose_cascade != null) {
			nose_cascade.detectMultiScale(grayscaleImage, potentialNoses, 1.1, 2, 2,
					new Size((face.width() / 50), (face.height() / 50)), new Size());
		}

		Rect[] temp = potentialNoses.toArray();

//		//This section will draw a green dot in the center of the detected nose(s), if there are any
//		for (int i = 0; i <temp.length; i++){
//			Point center = new Point( (temp[i].tl().x + (temp[i].width/2)) ,
//					(temp[i].tl().y + (temp[i].height/2)));
//			Core.rectangle(face, center , center, new Scalar(0, 255, 0, 255), 3);
//		}

		Rect nose = new Rect(new Point(-1, -1), new Point(-1, -1));
		if (temp.length != 0)
			nose = temp[0];

		return nose;
	}

	public static Rect[] buildEyeArray(Mat face) {

		// Create a grayscale image
		Mat grayscaleImage;
		grayscaleImage = new Mat(face.height(), face.width(), CvType.CV_8UC4);

		Imgproc.cvtColor(face, grayscaleImage, Imgproc.COLOR_RGBA2RGB);

		MatOfRect eyes = new MatOfRect();

		CascadeClassifier eye_cascade = new CascadeClassifier();
		eye_cascade.load("Cascades/haarcascade_eye.xml");

		// Use the classifier to detect eyes
		if (eye_cascade != null) {
			eye_cascade.detectMultiScale(grayscaleImage, eyes, 1.1, 2, 2,
					new Size((face.width() / 50), (face.height() / 50)), new Size());
		}

		Rect[] eyesArray = eyes.toArray();

//		//This section will draw a green dot in the center of the detected eye(s), if there are any
//		for (int i = 0; i <eyesArray.length; i++){
//			Point center = new Point( (eyesArray[i].tl().x +
//					(eyesArray[i].width/2)) , (eyesArray[i].tl().y +
//							(eyesArray[i].height/2)));
//			Core.rectangle(face, center , center, new Scalar(0, 255, 0, 255), 3);
//		}

		return eyesArray;
	}

	/*
	 * Tested, works on CV_8UC4.
	 */
	public static Rect[] buildFaceArray(Mat aInputFrame) {

		// Create a grayscale image
		Mat grayscaleImage;
		grayscaleImage = new Mat(aInputFrame.height(), aInputFrame.width(), CvType.CV_8UC4);

		Imgproc.cvtColor(aInputFrame, grayscaleImage, Imgproc.COLOR_RGBA2RGB);

		MatOfRect faces = new MatOfRect();

		CascadeClassifier face_cascade = new CascadeClassifier();
		face_cascade.load("Cascades/haarcascade_frontalface_alt2.xml");

		// Use the classifier to detect faces
		if (face_cascade != null) {
			face_cascade.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2,
					new Size((aInputFrame.width() / 100), (aInputFrame.height() / 100)), new Size());
		}

		Rect[] facesArray = faces.toArray();

		return facesArray;
	}

	/*
	 * Does what it says on the tin. Converts a jpg to an OpenCV "Mat" and
	 * returns it.
	 * 
	 * NOTE: Make sure System.loadLibrary(Core.NATIVE_LIBRARY_NAME); has been
	 * called before using this.
	 * 
	 */
	public static Mat bufferedImageToMat(BufferedImage image) {

		Mat newMat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);

		byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		newMat.put(0, 0, pixels);
		return newMat;
	}

	/*
	 * Does what it says on the tin. Writes an OpenCV CvType.CV_8UC3 Mat to a
	 * .jpg
	 * 
	 * NOTE: Make sure System.loadLibrary(Core.NATIVE_LIBRARY_NAME); has been
	 * called before using this.
	 */
	public static void writeMatToJpgFile(Mat toWrite, String filename) {
		Highgui.imwrite(filename, toWrite);
	}

	/*
	 * Does what it says on the tin. Writes a BufferedImage to a .jpg.
	 * 
	 */
	public static void writeBuffImageToJpgFile(BufferedImage toWrite, String filename) throws IOException {
		File outputfile = new File(filename);
		ImageIO.write(toWrite, "jpg", outputfile);
	}

	/**
	 * Align the face so that the eyes are perfectly level. If they are already
	 * aligned, do nothing.
	 */
	public static Mat alignEyes(Mat image, Point eyeA, Point eyeB) {
		double deltaY = eyeB.y - eyeA.y; // Change in Y coord
		if (deltaY == 0) { // If eyes are already aligned...
			return image;
		} else {
			double deltaX = eyeB.x - eyeA.x;// Change in X coord
			double arctan;
			if (deltaX == 0) // If eyes are perfectly vertical...
			{
				arctan = (Math.PI / 2); // (to avoid dividing by zero)
			} else {
				arctan = Math.atan(deltaY / deltaX);
			}

			Mat rotationMatrix = Imgproc.getRotationMatrix2D(new Point(image.width() / 2, image.height() / 2),
					Math.toDegrees(arctan), 1.0);

			Mat result = image.clone();
			Imgproc.warpAffine(image, result, rotationMatrix, result.size());
			return result;
		}
	}
}