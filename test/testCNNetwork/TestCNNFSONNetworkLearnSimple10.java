package testCNNetwork;

import static org.junit.Assert.*;

import java.util.LinkedList;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.FSONNetwork;
import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNFSONNetworkLearnSimple10 {

	LinkedList<Layer> layers;
	Cell[] out;
	double[] expect;

	@Before
	public void setUp() throws Exception {

		// Create the list of layers that will make up our network
		layers = new LinkedList<Layer>();
		
		Layer l0 = new Layer(4,4,1,1,1,1,16,1,0, LayerType.LOCAL);
		l0.initLayer();
		
		Layer l1 = new Layer(4,4,1,1,1,1,1,1,0, LayerType.CONV);
		l1.initLayer();
		
		Layer l2 = new Layer(4,4,1,2,2,1,4,2,0, LayerType.LOCAL);
		l2.initLayer();

		// Create and initialize the second layer
		Layer l3 = new Layer(2, 2, 1, 2, 2, 1, 2, 1, 0, LayerType.FULLY);
		l3.initLayer();

		// Add the layers to the list
		layers.add(0, l0);
		layers.add(1, l1);
		layers.add(2, l2);
		layers.add(3, l3);

		// Create and initialize the array of Cell that will hold the output of
		// this network.
		out = new Cell[2];
		out[0] = new Cell();
		out[1] = new Cell();

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/1b.jpg");
		
		// Feed the input through the layers of the network.
		// This sets up the connections needed for backpropagation
		FSONNetwork.feedForward(layers, out, true);
		
		// Here we are using the sigmoid activation function because 
		// the output cells are independant of one another. 
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);
		
		double test[][][]= new double[layers.getLast().depth][layers.getLast().rows][layers.getLast().collumns];
		// Depth
		for (int l = 0; l < layers.getLast().depth; l++) {
			// Row
			for (int j = 0; j < layers.getLast().rows; j++) {
				// Column
				for (int k = 0; k < layers.getLast().collumns; k++) {
					test[l][j][k]= layers.getLast().cells[l][j][k].value;
				}
			}
		}
		
		// Set up the input array (an array of the filenames of the input files,
		// located from the root directory of the project).
		String[] input = new String[12];
		input[0] = "testingInput/1b.jpg";
		input[1] = "testingInput/2b.jpg";
		input[2] = "testingInput/3b.jpg";
		input[3] = "testingInput/4b.jpg";
		input[4] = "testingInput/5b.jpg";
		input[5] = "testingInput/6b.jpg";
		input[6] = "testingInput/7b.jpg";
		input[7] = "testingInput/8b.jpg";
		input[8] = "testingInput/9b.jpg";
		input[9] = "testingInput/10b.jpg";
		input[10] = "testingInput/1.jpg";
		input[11]= "testingInput/6.jpg";

		// Set up the "dictionary". This is an array of the expected outputs
		// for each input.
		double[][] dictionary = new double[input.length][out.length];
		dictionary[0][0] = 1.0;
		dictionary[1][0] = 1.0;
		dictionary[2][0] = 1.0;
		dictionary[3][0] = 1.0;
		dictionary[4][0] = 0.0;
		dictionary[5][0] = 0.0;
		dictionary[6][0] = 0.0;
		dictionary[7][0] = 0.0;
		dictionary[8][0] = 0.0;
		dictionary[9][0] = 0.0;
		dictionary[10][0]= 1.0;
		dictionary[11][0] =0.0;

		dictionary[0][1] = 0.0;
		dictionary[1][1] = 0.0;
		dictionary[2][1] = 0.0;
		dictionary[3][1] = 1.0;
		dictionary[4][1] = 1.0;
		dictionary[5][1] = 1.0;
		dictionary[6][1] = 0.0;
		dictionary[7][1] = 1.0;
		dictionary[8][1] = 0.0;
		dictionary[9][1] = 0.0;
		dictionary[10][1] = 0.0;
		dictionary[11][1] = 1.0;
		
		// Actually call the learning function.
		FSONNetwork.learn(1, layers, out, input, 1500, dictionary, true, "TestCNNFSONNetworkLearnSimple8");
	}

	/**
	 * @throws java.lang.Exception
	 */
	@Test
	public void test() throws Exception {

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/1b.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because 
		// the output cells are independant of one another. 
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);
		
		assertEquals( 1.0, out[0].value, 0.1);
		assertEquals( 0.0, out[1].value, 0.1);
		
		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/7b.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.1);
		assertEquals(0.0, out[1].value, 0.1);

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/8b.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.1);
		assertEquals(1.0, out[1].value, 0.1);
		
		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/4.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.1);
		assertEquals(0.0, out[1].value, 0.1);

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/8.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.1);
		assertEquals(0.0, out[1].value, 0.1);
		
		FSONNetwork.openFileInputBW(layers, "testingInput/9.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(1.0, out[0].value, 0.1);
		assertEquals(1.0, out[1].value, 0.1);		
		
		FSONNetwork.openFileInputBW(layers, "testingInput/7.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.1);
		assertEquals(1.0, out[1].value, 0.1);
		
		FSONNetwork.openFileInputBW(layers, "testingInput/10.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(1.0, out[0].value, 0.1);
		assertEquals(1.0, out[1].value, 0.1);
	}

}
