package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.Layer;

/**
 * This tests the softmax() function in Layer.java, and, by extension, the
 * softmaxActivationFunction and sumE functions.
 *
 */
public class TestCNNSoftmax {

	Cell input[];//This will hold our input
	
	@Before
	public void setUp() throws Exception {
		//Set up an array to use as input
		input = new Cell[10];

		// Initialize the cells because java won't do it for you
		for (int i = 0; i < 10; i++) {
			input[i] = new Cell();
		}

		input[0].value = 80;
		input[1].value = 86;
		input[2].value = 24;
		input[3].value = 92;
		input[4].value = 89;
		input[5].value = 39;
		input[6].value = 79;
		input[7].value = 93;
		input[8].value = 5;
		input[9].value = 14;
		
	}

	@Test
	public void test() throws Exception {
		Layer.softmax(input);//Calculate the softmax for the input array
		
		//These are the values we expect
		double[] testout = {0.0000016295242044077288, 0.0006573969837514191, 7.790580672972041E-31, 0.2652128720005982, 0.013204171390331334, 2.546754356159107E-24, 0.0000005994684536928545, 0.7209233306326609, 4.364903764087444E-39, 3.5369181536190267E-35};
		
		//Because this is a softmax, the values should sum to 1. 
		//To test this, we will sum the values of output using this variable.
		double sum = 0;
		
		//Iterate over the output, testing that we have the expected values, and summing the total of the output values.
		for(int i = 0; i< input.length; i++){
			sum += input[i].value;//Add this value to the sum
			assertEquals(testout[0], input[0].value, 0);//Test to make sure this value is as expected
		}
		
		//Test that the sum of the values is 1, within a tolerance of >0.00001
		assertEquals(1, sum, 0.00001);
	}

}
