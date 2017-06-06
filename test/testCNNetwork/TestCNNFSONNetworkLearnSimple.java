package testCNNetwork;

import static org.junit.Assert.assertEquals;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedList;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.FSONNetwork;
import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNFSONNetworkLearnSimple {
	LinkedList<Layer> layers;
	Cell[] out;
	double[] expect;

	@Before
	public void setUp() throws Exception {

		// Create the list of layers that will make up our network
		layers = new LinkedList<Layer>();

		// Create and initialize the first layer
		Layer l1 = new Layer(2, 2, 1, 2, 2, 1, 1, 1, 0, LayerType.FULLY);
		l1.initLayer();

		// Add the second layer to the list
		layers.add(0, l1);

		// Create and initialize the array of Cell that will hold the output of
		// this network.
		out = new Cell[1];
		out[0] = new Cell();

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/1b.jpg");
		
		// Feed the input through the layers of the network.
		// This sets up the connections needed for backpropagation
		FSONNetwork.feedForward(layers, out, true);
		
		// Since network has a single output value, it is independent and needs
		// to use the regular sigmoid activation function instead of the softmax
		// (since the softmax would always result in a value of 1.0).
		out[0].value = Layer.activationFunction(out[0].value);

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
		input[10] = "testingInput/11b.jpg";
		input[11] = "testingInput/0b.jpg";

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
		dictionary[10][0] = 0;
		dictionary[11][0] = 0;
	
		PrintStream stdout = System.out;
		PrintStream outStream = new PrintStream(new FileOutputStream("simple1.csv"));
		System.setOut(outStream);
		
		long startTime = System.nanoTime();
		
		FSONNetwork.learn(1, layers, out, input, 500, dictionary, true, "TestCNNFSONNetworkLearnSimple1");
		
		long endTime = System.nanoTime();
		long duration = (endTime - startTime)/1000000;
		
		System.setOut(stdout);
		
		System.out.println(duration);
		
		outStream.close();
	}

	/**
	 * @throws java.lang.Exception
	 */
	@Test
	public void test() throws Exception {
		double[] results= new double[5];
		double[] expected= new double[5];
		
		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/3.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because 
		// the output cells are independant of one another. 
		out[0].value = Layer.activationFunction(out[0].value);
		
		expected[0] = 1.0;
		results[0] = out[0].value;

		assertEquals( 1.0, out[0].value, 0.2);

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/4.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);

		expected[1] = 0.0;
		results[1] = out[0].value;
		
		assertEquals(0.0, out[0].value, 0.2);

		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/7.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);
		
		expected[2] = 0.0;
		results[2] = out[0].value;
		
		assertEquals(0.0, out[0].value, 0.2);
		
		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/9.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);

		expected[3] = 1.0;
		results[3] = out[0].value;
		
		assertEquals(1.0, out[0].value, 0.2);
		
		// Open file for input
		FSONNetwork.openFileInputBW(layers, "testingInput/8.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(layers, out, false);
		// Here we are using the sigmoid activation function because
		// the output cells are independant of one another.
		out[0].value = Layer.activationFunction(out[0].value);

		expected[4] = 0.0;
		results[4] = out[0].value;

		assertEquals(0.0, out[0].value, 0.2);

		System.out.println(Arrays.toString(results));
		System.out.println(Arrays.toString(expected));
		double testError = FSONNetwork.crossEntropyTotalErrorArray(results, expected);

		PrintStream resultStream = new PrintStream(new FileOutputStream("simple1test.csv"));
		resultStream.print(testError);
		resultStream.close();
	}
}