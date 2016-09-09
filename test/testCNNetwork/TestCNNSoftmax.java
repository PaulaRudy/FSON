package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;

/**
 * This tests the softmax() function in Layer.java, and, by extension, the
 * softmaxActivationFunction and sumE functions.
 *
 */
public class TestCNNSoftmax {

	double[] input;//This will hold our input
	
	@Before
	public void setUp() throws Exception {
		//Set up an array to use as input
		input = new double[10];
		input[0] = 80;
		input[1] = 86;
		input[2] = 24;
		input[3] = 92;
		input[4] = 89;
		input[5] = 39;
		input[6] = 79;
		input[7] = 93;
		input[8] = 5;
		input[9] = 14;
		
	}

	@Test
	public void test() throws Exception {
		double[] output = Layer.softmax(input);//Calculate the softmax for the input array
		
		//These are the values we expect
		double[] testout = {0.000001629532314185, 0.0006574006246691, 7.790591753178105E-31, 0.26521441687369, 0.013204245350262, 2.5467605009689E-24, 0.0000005994713707259, 0.720927577174, 4.394315424468E-39, 3.536923173694E-35};
		
		//Because this is a softmax, the values should sum to 1. 
		//To test this, we will sum the values of output using this variable.
		double sum = 0;
		
		//Iterate over the output, testing that we have the expected values, and summing the total of the output values.
		for(int i = 0; i< output.length; i++){
			sum += output[i];//Add this value to the sum
			assertEquals(testout[0], output[0], 0);//Test to make sure this value is as expected
		}
		
		//Test that the sum of the values is 1, within a tolerance of >0.00001
		assertEquals(1, sum, 0.00001);
	}

}
