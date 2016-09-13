package testCNNetwork;

import static org.junit.Assert.assertEquals;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.junit.Before;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import cnnetwork.Cell;
import cnnetwork.FSONNetwork;
import cnnetwork.Filter;
import cnnetwork.Layer;
import cnnetwork.LayerType;
import image.Align;

/**
 * This tests the functions declared in the cnnetwork.FSONNetwork.java file.
 * This is to ensure proper functionality if the functions are modified.
 *
 */
public class TestCNNFSONnetwork {
	
	Layer l1, l2, l3, l4, l5, l6, l7, l8;
	Cell[] out;
	BufferedImage image;

	@Before
	public void setUp() throws Exception {
		
		//The file named here must be in same location as Align class file
		image = ImageIO.read(Align.class.getResource("print.jpg"));

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
		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here in the order RGB
		Core.split(resizedImage, channels);

		//Declare and initialize the first layer
		Layer l1 = new Layer(76, 76, 3, 5, 5, 3, 32, 1, 0, LayerType.CONV);

		//Create and initialize the filters
		for (int i = 0; i < l1.K; i++) {
			//Create the filter weights
			double[][][] newFilterWeights = new double[l1.Fdepth][l1.Frows][l1.Fcollumns];
			for (int x = 0; x < l1.Fdepth; x++) {
				for (int y = 0; y < l1.Frows; y++) {
					for (int z = 0; z < l1.Fcollumns; z++) {
						newFilterWeights[x][y][z] = 1; //Because this is a simple test, we're going to set every entry in the filter to 1.
					}
				}
			}
			
			Filter newFilter = new Filter(newFilterWeights);//Use the default constructor with the newly created filter weights
			l1.filters.add(newFilter);//Actually add the filter to the list of filters in the layer
			double newBias = 1;//Create the bias for this filter
			l1.biases.add(newBias);//Add the bias to the list of biases in the layer.
		}

		//Create the second layer.
		Layer l2 = new Layer(72, 72, 32, 3, 3, 32, 39200, 2, 0, LayerType.MAXPOOL);
		
		// Create and initialize the "filters"
		// Because this is a maxpool layer, these filters are only used to
		// record connections for use during backpropagation, and we don't need
		// any biases.
		for (int i = 0; i < l2.K; i++) {
			// Create the filter weights
			double[][][] newFilterWeights = new double[1][l2.Frows][l2.Fcollumns];

			// (java will initialize them to 0)

			Filter newFilter = new Filter(newFilterWeights);// Use the default constructor with the newly created filter weights
			l2.filters.add(newFilter);// Actually add the filter to the list of filters in the layer
		}
		
		//Create and initialize the third layer.
		Layer l3 = new Layer(35, 35, 32, 5, 5, 32, 16, 1, 0, LayerType.CONV);

		//Create and initialize the filters
		for (int i = 0; i < l3.K; i++) {
			//Create the filter weights
			double[][][] newFilterWeights = new double[l3.Fdepth][l3.Frows][l3.Fcollumns];
			for (int x = 0; x < l3.Fdepth; x++) {
				for (int y = 0; y < l3.Frows; y++) {
					for (int z = 0; z < l3.Fcollumns; z++) {
						newFilterWeights[x][y][z] = 0.5;//Here we are setting every entry in the filter to 0.5.
					}
				}
			}

			Filter newFilter = new Filter(newFilterWeights);//Use the default constructor with the newly created filter weights
			l3.filters.add(newFilter);//Actually add the filter to the list of filters in the layer
			double newBias = 1;//Create the bias for this filter
			l3.biases.add(newBias);//Add the bias to the list of biases in the layer.
		}

		//Create the fourth layer.
		Layer l4 = new Layer(31, 31, 16, 5, 5, 16, 3136, 2, 0, LayerType.MAXPOOL);

		// Create and initialize the "filters"
		// Because this is a maxpool layer, these filters are only used to
		// record connections for use during backpropagation, and we don't need
		// any biases.
		for (int i = 0; i < l4.K; i++) {
			// Create the filter weights
			double[][][] newFilterWeights = new double[1][l4.Frows][l4.Fcollumns];

			// (java will initialize them to 0)

			Filter newFilter = new Filter(newFilterWeights);// Use the default constructor with the newly created filter weights
			l4.filters.add(newFilter);// Actually add the filter to the list of filters in the layer
		}
		
		//Create and initialize the fifth layer.
		Layer l5 = new Layer(14, 14, 16, 3, 3, 16, 16, 2, 0, LayerType.CONV);

		//Create and initialize the filters
		for (int i = 0; i < l5.K; i++) {
			//Create the filter weights
			double[][][] newFilterWeights = new double[l5.Fdepth][l5.Frows][l5.Fcollumns];
			for (int x = 0; x < l5.Fdepth; x++) {
				for (int y = 0; y < l5.Frows; y++) {
					for (int z = 0; z < l5.Fcollumns; z++) {
						newFilterWeights[x][y][z] = 1;//Here we are setting every entry in the filter to 1.
					}
				}
			}

			Filter newFilter = new Filter(newFilterWeights);//Use the default constructor with the newly created filter weights
			l5.filters.add(newFilter);//Actually add the filter to the list of filters in the layer
			double newBias = 1;//Create the bias for this filter
			l5.biases.add(newBias);//Add the bias to the list of biases in the layer.
		}

		//Create and initialize the sixth layer.
		//This one is a locally connected layer.
		Layer l6 = new Layer(6, 6, 16, 3, 3, 1, 400, 1, 0, LayerType.LOCAL);

		//Create and initialize the filters
		for (int i = 0; i < l6.K; i++) {
			//Create the filter weights
			double[][][] newFilterWeights = new double[l6.Fdepth][l6.Frows][l6.Fcollumns];
			for (int x = 0; x < l6.Fdepth; x++) {
				for (int y = 0; y < l6.Frows; y++) {
					for (int z = 0; z < l6.Fcollumns; z++) {
						newFilterWeights[x][y][z] = 0.5;//Here we are setting every entry in the filter to 0.5.
					}
				}
			}

			Filter newFilter = new Filter(newFilterWeights);//Use the default constructor with the newly created filter weights
			l6.filters.add(newFilter);//Actually add the filter to the list of filters in the layer
			double newBias = 1;//Create the bias for this filter
			l6.biases.add(newBias);//Add the bias to the list of biases in the layer.
		}

		//Create and initialize the seventh layer.
		//This one is also a locally connected layer.
		Layer l7 = new Layer(5, 5, 16, 5, 5, 16, 2048, 1, 0, LayerType.FULLY);

		//Create and initialize the filters
		for (int i = 0; i < l7.K; i++) {
			//Create the filter weights
			double[][][] newFilterWeights = new double[l7.Fdepth][l7.Frows][l7.Fcollumns];
			for (int x = 0; x < l7.Fdepth; x++) {
				for (int y = 0; y < l7.Frows; y++) {
					for (int z = 0; z < l7.Fcollumns; z++) {
						newFilterWeights[x][y][z] = 1;//Here we are setting every entry in the filter to 1.
					}
				}
			}

			Filter newFilter = new Filter(newFilterWeights);//Use the default constructor with the newly created filter weights
			l7.filters.add(newFilter);//Actually add the filter to the list of filters in the layer
			double newBias = 1;//Create the bias for this filter
			l7.biases.add(newBias);//Add the bias to the list of biases in the layer.
		}

		//Create and initialize the eighth layer.
		//This one is a fully connected layer.
		Layer l8 = new Layer(2048, 1, 1, 2048, 1, 1, 2016, 1, 0, LayerType.FULLY);

		//Create and initialize the filters
		for (int i = 0; i < l8.K; i++) {
			//Create the filter weights
			double[][][] newFilterWeights = new double[l8.Fdepth][l8.Frows][l8.Fcollumns];
			for (int x = 0; x < l8.Fdepth; x++) {
				for (int y = 0; y < l8.Frows; y++) {
					for (int z = 0; z < l8.Fcollumns; z++) {
						newFilterWeights[x][y][z] = 0.5;//Here we are setting every entry in the filter to 0.5.
					}
				}
			}

			Filter newFilter = new Filter(newFilterWeights);//Use the default constructor with the newly created filter weights
			l8.filters.add(newFilter);//Actually add the filter to the list of filters in the layer
			double newBias = 1;//Create the bias for this filter
			l8.biases.add(newBias);//Add the bias to the list of biases in the layer.
		}

		//This is the last "layer": this will hold the output of the network
		out = new Cell[2016];
		
		// Initialize the cells because java won't do it for you
		for (int i = 0; i < 2016; i++) {
			this.out[i] = new Cell();
		}
		
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
					l1.cells[i][j][k].value = temp[(j * 76) + k];
				}

			}

		}

		//Call the appropriate functions to feed the input through the layers
		l1.convolution(l1.cells, l1.filters, l2.cells, l1.step, l1.pad, l1.biases);
		l2.pool(l2.cells, l2.filters, l3.cells, l2.step, l2.Fcollumns);
		l3.convolution(l3.cells, l3.filters, l4.cells, l3.step, l3.pad, l3.biases);
		l4.pool(l4.cells, l4.filters, l5.cells, l4.step, l4.Fcollumns);
		l5.convolution(l5.cells, l5.filters, l6.cells, l5.step, l5.pad, l5.biases);
		l6.local(l6.cells, l6.filters, l7.cells, l6.step, l6.pad, l6.biases);
		l7.full(l7.cells, l7.filters, l8.cells[0][0], l7.step, l7.pad, l7.biases);
		l8.full(l8.cells, l8.filters, out, l8.step, l8.pad, l8.biases);
		Layer.softmax(out);
	}

	@Test
	public void test() throws Exception {
		FSONNetwork test = new FSONNetwork();
		
		test.calculate(image);
		
		//Test that using the functions in FSONNetwork results in the same values as doing it manually
		for (int i = 0; i < out.length; i++) {
			assertEquals(out[i].value, test.out[i].value, 0);
		}
	}

}
