package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.FSONNetwork;
import cnnetwork.Layer;

public class TestCNNFSONNetworkLearnSimpleColors {

	FSONNetwork colorNet;

	@Before
	public void setUp() throws Exception {
		colorNet = FSONNetwork.colorNetwork();

		// This array will hold the filenames of the pictures of each input file
		String[] learnInput = new String[18];

		learnInput[0] = "testingInput/colors/red/0.jpg";
		learnInput[1] = "testingInput/colors/orange/0.jpg";
		learnInput[2] = "testingInput/colors/yellow/0.jpg";
		learnInput[3] = "testingInput/colors/green/0.jpg";
		learnInput[4] = "testingInput/colors/aqua/0.jpg";
		learnInput[5] = "testingInput/colors/blue/0.jpg";
		learnInput[6] = "testingInput/colors/purple/0.jpg";
		learnInput[7] = "testingInput/colors/pink/0.jpg";
		learnInput[8] = "testingInput/colors/red/1.jpg";
		learnInput[9] = "testingInput/colors/red/2.jpg";
		learnInput[10] = "testingInput/colors/orange/1.jpg";
		learnInput[11] = "testingInput/colors/yellow/1.jpg";
		learnInput[12] = "testingInput/colors/green/1.jpg";
		learnInput[13] = "testingInput/colors/aqua/1.jpg";
		learnInput[14] = "testingInput/colors/blue/1.jpg";
		learnInput[15] = "testingInput/colors/purple/1.jpg";
		learnInput[16] = "testingInput/colors/pink/1.jpg";
		learnInput[17] = "testingInput/colors/red/3.jpg";

		// This is the dictionary used to tell the learning functions what the
		// ideal
		// output for a picture of that person would look like
		double[][] dictionary = new double[18][9];

		// Given an index, "dictionary[x][y]",
		// x is the index in the list of names of the person the input
		// represents, and y is the array of output we would expect to
		// see in a perfectly trained network.
		dictionary[0][0] = 1.0;
		dictionary[1][1] = 1.0;
		dictionary[2][2] = 1.0;
		dictionary[3][3] = 1.0;
		dictionary[4][4] = 1.0;
		dictionary[5][5] = 1.0;
		dictionary[6][6] = 1.0;
		dictionary[7][7] = 1.0;
		dictionary[8][8] = 1.0;
		dictionary[9][0] = 1.0;
		dictionary[10][1] = 1.0;
		dictionary[11][2] = 1.0;
		dictionary[12][3] = 1.0;
		dictionary[13][4] = 1.0;
		dictionary[14][5] = 1.0;
		dictionary[15][6] = 1.0;
		dictionary[16][7] = 1.0;
		dictionary[17][8] = 1.0;

		FSONNetwork.openHSVFileInput(colorNet.layers, learnInput[0]);

		FSONNetwork.feedForward(colorNet.layers, colorNet.out, true);

		// Use the learning function to learn using our newly processed input
		// and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		// The learning factor is set low because the input ranges from 0 to 1.
		FSONNetwork.learn(0.225, colorNet.layers, colorNet.out, learnInput, 5000, dictionary, false,
				"testlearncolor.txt");
	}

	@Test
	public void test() throws Exception {
		
		// Open file for input
		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/blue.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] blueTest = new double[colorNet.out.length];

		int maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			blueTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue(maxIndex == 5);
		assertTrue(colorNet.out[maxIndex].value > 0.3);

		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/red.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] redTest = new double[colorNet.out.length];

		maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			redTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue((maxIndex == 0) || (maxIndex == 8));
		assertTrue(colorNet.out[maxIndex].value > 0.3);

		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/yellow.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] yellowTest = new double[colorNet.out.length];

		maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			yellowTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue(maxIndex == 2);
		assertTrue(colorNet.out[maxIndex].value > 0.3);

		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/purple.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] purpleTest = new double[colorNet.out.length];

		maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			purpleTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue(maxIndex == 6);
		assertTrue(colorNet.out[maxIndex].value > 0.3);

		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/green.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] greenTest = new double[colorNet.out.length];

		maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			greenTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue(maxIndex == 3);
		assertTrue(colorNet.out[maxIndex].value > 0.3);

		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/aqua.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] aquaTest = new double[colorNet.out.length];

		maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			aquaTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue(maxIndex == 4);
		assertTrue(colorNet.out[maxIndex].value > 0.3);

		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/pink.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] pinkTest = new double[colorNet.out.length];

		maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			pinkTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue(maxIndex == 7);
		assertTrue(colorNet.out[maxIndex].value > 0.3);

		FSONNetwork.openHSVFileInput(colorNet.layers, "testingInput/colors/orange.jpg");
		// Feed the input through the layers of the network.
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, false);
		Layer.softmax(colorNet.out);

		// This array allows for easy viewing when debugging
		double[] orangeTest = new double[colorNet.out.length];

		maxIndex = 0;

		for (int i = 0; i < colorNet.out.length; i++) {
			orangeTest[i] = colorNet.out[i].value;
			if (colorNet.out[i].value > colorNet.out[maxIndex].value) {
				maxIndex = i;
			}
		}

		assertTrue(maxIndex == 1);
		assertTrue(colorNet.out[maxIndex].value > 0.3);

	}

}
