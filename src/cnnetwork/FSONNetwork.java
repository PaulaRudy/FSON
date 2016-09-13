package cnnetwork;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import image.Align;

/**
 * This class is a simple demonstration of the FSON system.
 * 
 * @author Paula Rudy
 *
 */
public class FSONNetwork {

	public LinkedList<Layer> layers; //The layers that make up this sample network
	public Cell[] out; //This is the last layer of this network, the "output".
	
	/**
	 * This function creates and sets up the network. 
	 * There are 8 layers and an output layer.
	 * Layers 1,3, and 5 are all convoltional layers.
	 * Layers 2 and 4 are maxpool layers.
	 * Layers 6 and 7 are locally connected layers,
	 * and layer 8 is a fully connected layer.
	 * The weights of all the filters are either 1 or 0.5 depending
	 * on what layer they are for.
	 * 
	 */
	public FSONNetwork() {

		//Declare and initialize the first layer
		Layer l1 = new Layer(76, 76, 3, 5, 5, 3, 32, 1, 0, LayerType.CONV);

		//Setup the layer. This creates and initializes the filters and biases, all with a value of 1.
		l1.initLayer();

		//Create the second layer.
		Layer l2 = new Layer(72, 72, 32, 3, 3, 32, 39200, 2, 0, LayerType.MAXPOOL);
		l2.initLayer();
		
		//Create and initialize the third layer.
		Layer l3 = new Layer(35, 35, 32, 5, 5, 32, 16, 1, 0, LayerType.CONV);
		l3.initLayer();

		//Create the fourth layer.
		Layer l4 = new Layer(31, 31, 16, 5, 5, 16, 3136, 2, 0, LayerType.MAXPOOL);
		l4.initLayer();
		
		//Create and initialize the fifth layer.
		Layer l5 = new Layer(14, 14, 16, 3, 3, 16, 16, 2, 0, LayerType.CONV);
		l5.initLayer();

		//Create and initialize the sixth layer.
		//This one is a locally connected layer.
		Layer l6 = new Layer(6, 6, 16, 3, 3, 1, 400, 1, 0, LayerType.LOCAL);
		l6.initLayer();

		//Create and initialize the seventh layer.
		//This one is also a locally connected layer.
		Layer l7 = new Layer(5, 5, 16, 5, 5, 16, 2048, 1, 0, LayerType.FULLY);
		l7.initLayer();

		//Create and initialize the eighth layer.
		//This one is a fully connected layer.
		Layer l8 = new Layer(2048, 1, 1, 2048, 1, 1, 2016, 1, 0, LayerType.FULLY);
		l8.initLayer();

		//This is the last "layer": this will hold the output of the network
		this.out = new Cell[2016];
		
		//Initialize the cells because java won't do it for you
		for (int i = 0; i < 2016; i++) {
			this.out[i] = new Cell();
		}

		//Initialize the list of layers
		this.layers = new LinkedList<Layer>();

		//Add each layer at the appropriate place in the list.
		this.layers.add(0, l1);
		this.layers.add(1, l2);
		this.layers.add(2, l3);
		this.layers.add(3, l4);
		this.layers.add(4, l5);
		this.layers.add(5, l6);
		this.layers.add(6, l7);
		this.layers.add(7, l8);

	}

	/**
	 * This function feeds a buffered image through the network,
	 * 
	 * @param image
	 *            The image to use as input
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 *             (see Layer::activationFunction()).
	 */
	public void calculate(BufferedImage image) throws Exception {

		//This is necessary to use any of the OpenCV functions
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		//First, convert the image to an OpenCV Mat.
		//This function can be found in image.Align.java
		Mat test = Align.bufferedImageToMat(image);

		//Next, resize the image to the size needed.
		Mat resizedImage = new Mat();
		Size sz = new Size(76, 76);
		Imgproc.resize(test, resizedImage, sz);

		//Split the image into three color channels
		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here
													// in the order RGB
		Core.split(resizedImage, channels);

		//For each channel...
		for (int i = 0; i < channels.size(); i++) {

			//Feed the individual pixel values into a temporary array...
			channels.get(i).convertTo(channels.get(i), CvType.CV_64FC3);
			int size = (int) (channels.get(i).total() * channels.get(i).channels());
			double[] temp = new double[size];
			channels.get(i).get(0, 0, temp);
			
			//...and then into the cells of the first layer of the network.
			for (int j = 0; j < 76; j++) {
				for (int k = 0; k < 76; k++) {
					this.layers.get(0).cells[i][j][k].value = temp[(j * 76) + k];
				}

			}

		}

		//Call the appropriate functions to feed the input through the layers
		this.layers.get(0).convolution(this.layers.get(0).cells, this.layers.get(0).filters, this.layers.get(1).cells, this.layers.get(0).step, this.layers.get(0).pad, this.layers.get(0).biases);
		this.layers.get(1).pool(this.layers.get(1).cells, this.layers.get(1).filters, this.layers.get(2).cells, this.layers.get(1).step, this.layers.get(1).Fcollumns);
		this.layers.get(2).convolution(this.layers.get(2).cells, this.layers.get(2).filters, this.layers.get(3).cells, this.layers.get(2).step, this.layers.get(2).pad, this.layers.get(2).biases);
		this.layers.get(3).pool(this.layers.get(3).cells, this.layers.get(3).filters, this.layers.get(4).cells, this.layers.get(3).step, this.layers.get(3).Fcollumns);
		this.layers.get(4).convolution(this.layers.get(4).cells, this.layers.get(4).filters, this.layers.get(5).cells, this.layers.get(4).step, this.layers.get(4).pad, this.layers.get(4).biases);
		this.layers.get(5).local(this.layers.get(5).cells, this.layers.get(5).filters, this.layers.get(6).cells, this.layers.get(5).step, this.layers.get(5).pad, this.layers.get(5).biases);
		this.layers.get(6).full(this.layers.get(6).cells, this.layers.get(6).filters, this.layers.get(7).cells[0][0], this.layers.get(6).step, this.layers.get(6).pad, this.layers.get(6).biases);
		this.layers.get(7).full(this.layers.get(7).cells, this.layers.get(7).filters, this.out, this.layers.get(7).step, this.layers.get(7).pad, this.layers.get(7).biases);
		Layer.softmax(this.out);
	}
	
	

}
