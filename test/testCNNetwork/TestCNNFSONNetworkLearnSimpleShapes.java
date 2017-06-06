package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.FSONNetwork;
import cnnetwork.Layer;

public class TestCNNFSONNetworkLearnSimpleShapes {

	FSONNetwork shapeNet;
	@Before
	public void setUp() throws Exception {
		shapeNet = FSONNetwork.shapeNetwork();

		// This array will hold the filenames of the pictures of each input file
		String[] learnInput = new String[832];

		for (int i = 0; i< 416; i++){
			learnInput[i] = "testingInput/circle/"+ (i)+ ".jpg";
		}
		
		for (int i = 416; i< 832; i++){
			learnInput[i] = "testingInput/circle/"+ (i)+ ".jpg";
		}
		
		// This is the dictionary used to tell the learning functions what the
		// ideal output for a picture of that person would look like
		double[][] dictionary = new double[832][1];

		// Given an index, "dictionary[x][y]",
		// x is the index in the list of input,
		// and y is the array of output we would expect to
		// see in a perfectly trained network.
		
		for (int i = 0; i< 416; i++){
			dictionary[i][0] = 1.0;
		}
		
		for (int i = 416; i< 832; i++){
			dictionary[i][0] = 0.0;
		}
	
		FSONNetwork.openFileInputBW(shapeNet.layers, learnInput[0]);

		FSONNetwork.feedForward(shapeNet.layers, shapeNet.out, true);

		// Use the learning function to learn using our newly processed input
		// and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		// The learning factor is set low because the input ranges from 0 to 1.
		FSONNetwork.learn(0.3, shapeNet.layers, shapeNet.out, learnInput, 3000, dictionary, true,
				"testlearnshape.txt");
		
	}

	@Test
	public void test() throws Exception {
		// Open file for input
		FSONNetwork.openFileInputBW(shapeNet.layers, "testingInput/circle/test1.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(shapeNet.layers, shapeNet.out, false);
		double test1 = Layer.activationFunction(shapeNet.out[0].value);

//		assertEquals(1.0, shapeNet.out[0].value, 0.3);

		// Open file for input
		FSONNetwork.openFileInputBW(shapeNet.layers, "testingInput/circle/test2.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(shapeNet.layers, shapeNet.out, false);
		double test2 = Layer.activationFunction(shapeNet.out[0].value);

//		assertEquals(1.0, shapeNet.out[0].value, 0.3);

		// Open file for input
		FSONNetwork.openFileInputBW(shapeNet.layers, "testingInput/circle/test3.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(shapeNet.layers, shapeNet.out, false);
		double test3 = Layer.activationFunction(shapeNet.out[0].value);

//		assertEquals(0.0, shapeNet.out[0].value, 0.3);

		// Open file for input
		FSONNetwork.openFileInputBW(shapeNet.layers, "testingInput/circle/test4.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(shapeNet.layers, shapeNet.out, false);
		double test4 = Layer.activationFunction(shapeNet.out[0].value);

		assertEquals(0.0, shapeNet.out[0].value, 0.3);
		
	}

}
