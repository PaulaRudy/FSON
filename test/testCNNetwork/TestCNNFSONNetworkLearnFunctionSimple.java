package testCNNetwork;

import java.util.LinkedList;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.FSONNetwork;
import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNFSONNetworkLearnFunctionSimple {
	LinkedList<Layer> layers;
	Cell[] out;
	double[] expect;

	@Before
	public void setUp() throws Exception {

		// Create the list of layers that will make up our network
		layers = new LinkedList<Layer>();

		// Create and initialize the second layer
		Layer l1 = new Layer(1, 1, 1, 1, 1, 1, 1, 1, 0, LayerType.FULLY);
		l1.initLayer();

		// Add the second layer to the list
		layers.add(0, l1);

		// Create and initialize the array of Cell that will hold the output of
		// this network.
		out = new Cell[1];
		out[0] = new Cell();

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "1a.jpg");
		
		// Feed the input through the layers of the network.
		// This sets up the connections needed for backpropagation
		FSONNetwork.feedForward(layers, out, true);
		
		// Since network has a single output value, it is independent and needs
		// to use the regular sigmoid activation function instead of the softmax
		// (since the softmax would always result in a value of 1.0).
		out[0].value = Layer.activationFunction(out[0].value);

		// Set up the input array (an array of the filenames of the input files,
		// located in the root directory of the project).
		String[] input = new String[2];
		input[0] = "1a.jpg";
		input[1] = "2a.jpg";

		// Set up the "dictionary". This is an array of the expected outputs
		// for each input.
		double[][] dictionary = new double[input.length][out.length];
		dictionary[0][0] = 1;
		dictionary[1][0] = 0;

		FSONNetwork.learn(1, layers, out, input, 100, dictionary, true);

	}

	/**
	 * @throws java.lang.Exception
	 */
	@Test
	public void test() throws Exception {

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "1a.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Since network has a single output value, it is independent and needs
		// to use the regular sigmoid activation function instead of the softmax
		// (since the softmax would always result in a value of 1.0).
		out[0].value = Layer.activationFunction(out[0].value);

	}

}