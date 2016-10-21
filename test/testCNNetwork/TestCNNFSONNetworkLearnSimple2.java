package testCNNetwork;

import static org.junit.Assert.*;

import java.util.LinkedList;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.FSONNetwork;
import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNFSONNetworkLearnSimple2 {
	LinkedList<Layer> layers;
	Cell[] out;
	double[] expect;

	@Before
	public void setUp() throws Exception {

		// Create the list of layers that will make up our network
		layers = new LinkedList<Layer>();

		// Create and initialize the second layer
		Layer l1 = new Layer(2, 2, 1, 2, 2, 1, 2, 1, 0, LayerType.FULLY);
		l1.initLayer();

		// Add the second layer to the list
		layers.add(0, l1);

		// Create and initialize the array of Cell that will hold the output of
		// this network.
		out = new Cell[2];
		out[0] = new Cell();
		out[1] = new Cell();

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "1b.jpg");
		
		// Feed the input through the layers of the network.
		// This sets up the connections needed for backpropagation
		FSONNetwork.feedForward(layers, out, true);
		
		// Here we are using the sigmoid activation function because 
		// the output cells are independant of one another. 
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);
		
		// Set up the input array (an array of the filenames of the input files,
		// located in the root directory of the project).
		String[] input = new String[10];
		input[0] = "1b.jpg";
		input[1] = "2b.jpg";
		input[2] = "3b.jpg";
		input[3] = "4b.jpg";
		input[4] = "5b.jpg";
		input[5] = "6b.jpg";
		input[6] = "7b.jpg";
		input[7] = "8b.jpg";
		input[8] = "9b.jpg";
		input[9] = "10b.jpg";

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
		FSONNetwork.learn(1, layers, out, input, 500, dictionary, true);

	}

	/**
	 * @throws java.lang.Exception
	 */
	@Test
	public void test() throws Exception {

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "1b.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because 
		// the output cells are independant of one another. 
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);
		
		assertEquals( 1.0, out[0].value, 0.2);
		assertEquals( 0.0, out[1].value, 0.2);
		
		// Open file for input
		FSONNetwork.openFileInputBW(layers, "7b.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		out[1].value = Layer.activationFunction(out[1].value);

		assertEquals(0.0, out[0].value, 0.2);
		assertEquals(0.0, out[1].value, 0.2);

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "8b.jpg");
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
