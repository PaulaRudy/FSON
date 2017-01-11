package testCNNetwork;

import static org.junit.Assert.*;

import java.util.LinkedList;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.FSONNetwork;
import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNFSONNetworkLearnSimple6 {

	LinkedList<Layer> layers;
	Cell[] out;
	double[] expect;

	@Before
	public void setUp() throws Exception {

		// Create the list of layers that will make up our network
		layers = new LinkedList<Layer>();
		
		Layer l0 = new Layer(4,4,1,2,2,1,4,2,0, LayerType.LOCAL);
		l0.initLayer();

		// Create and initialize the second layer
		Layer l1 = new Layer(2, 2, 1, 2, 2, 1, 2, 1, 0, LayerType.FULLY);
		l1.initLayer();

		// Add the layers to the list
		layers.add(0, l0);
		layers.add(1, l1);

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
		
		// Set up the input array (an array of the filenames of the input files,
		// located from the root directory of the project).
		String[] input = new String[10];
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

		// Set up the "dictionary". This is an array of the expected outputs
		// for each input.
		double[][] dictionary = new double[input.length][out.length];
		dictionary[0][0] = 1;
		dictionary[1][0] = 1;
		dictionary[2][0] = 1;
		dictionary[3][0] = 1;
		dictionary[4][0] = 0;
		dictionary[5][0] = 0;
		dictionary[6][0] = 0;
		dictionary[7][0] = 0;
		dictionary[8][0] = 0;
		dictionary[9][0] = 0;

		dictionary[0][1] = 0;
		dictionary[1][1] = 0;
		dictionary[2][1] = 0;
		dictionary[3][1] = 1;
		dictionary[4][1] = 1;
		dictionary[5][1] = 1;
		dictionary[6][1] = 0;
		dictionary[7][1] = 1;
		dictionary[8][1] = 0;
		dictionary[9][1] = 0;
		// Actually call the learning function.
		FSONNetwork.learn(1, layers, out, input, 500, dictionary, true, "TestCNNFSONNetworkLearnSimple2");
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
		
		assertEquals( 1.0, out[0].value, 0.2);
		assertEquals( 0.0, out[1].value, 0.2);
		
		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/7b.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.2);
		assertEquals(0.0, out[1].value, 0.2);

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/8b.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.2);
		assertEquals(1.0, out[1].value, 0.2);
				

	}

}
