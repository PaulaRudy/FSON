package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Filter;
import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests the "full" function found in cnnetwork.Layer.java
 * 
 */
public class TestCNNFullFunction {

	Layer testLayer;

	Filter testFilter0, testFilter1, testFilter2;

	double testBias0, testBias1, testBias2;

	double[] testOut;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(3, 3, 3, 3, 3, 3, 3, 1, 0, LayerType.CONV);

		//Set the values of all the cells
		testLayer.cells[0][0][0] = 2;
		testLayer.cells[0][0][1] = 0;
		testLayer.cells[0][0][2] = 2;

		testLayer.cells[0][1][0] = 2;
		testLayer.cells[0][1][1] = 2;
		testLayer.cells[0][1][2] = 2;

		testLayer.cells[0][2][0] = 0;
		testLayer.cells[0][2][1] = 1;
		testLayer.cells[0][2][2] = 0;

		testLayer.cells[1][0][0] = 1;
		testLayer.cells[1][0][1] = 0;
		testLayer.cells[1][0][2] = 2;

		testLayer.cells[1][1][0] = 0;
		testLayer.cells[1][1][1] = 2;
		testLayer.cells[1][1][2] = 0;

		testLayer.cells[1][2][0] = 2;
		testLayer.cells[1][2][1] = 1;
		testLayer.cells[1][2][2] = 2;

		testLayer.cells[2][0][0] = 1;
		testLayer.cells[2][0][1] = 2;
		testLayer.cells[2][0][2] = 0;

		testLayer.cells[2][1][0] = 2;
		testLayer.cells[2][1][1] = 2;
		testLayer.cells[2][1][2] = 1;

		testLayer.cells[2][2][0] = 0;
		testLayer.cells[2][2][1] = 1;
		testLayer.cells[2][2][2] = 1;
		
		//Create and initialize the first filter
		double[][][] testFilter0Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter0Weights[0][0][0] = 1;
		testFilter0Weights[0][0][1] = 1;
		testFilter0Weights[0][0][2] = 0;

		testFilter0Weights[0][1][0] = 0;
		testFilter0Weights[0][1][1] = 1;
		testFilter0Weights[0][1][2] = 0;

		testFilter0Weights[0][2][0] = 1;
		testFilter0Weights[0][2][1] = 0;
		testFilter0Weights[0][2][2] = 0;

		testFilter0Weights[1][0][0] = 1;
		testFilter0Weights[1][0][1] = -1;
		testFilter0Weights[1][0][2] = 1;

		testFilter0Weights[1][1][0] = 0;
		testFilter0Weights[1][1][1] = -1;
		testFilter0Weights[1][1][2] = 0;

		testFilter0Weights[1][2][0] = 0;
		testFilter0Weights[1][2][1] = -1;
		testFilter0Weights[1][2][2] = 0;

		testFilter0Weights[2][0][0] = -1;
		testFilter0Weights[2][0][1] = 1;
		testFilter0Weights[2][0][2] = 0;

		testFilter0Weights[2][1][0] = -1;
		testFilter0Weights[2][1][1] = 1;
		testFilter0Weights[2][1][2] = 0;

		testFilter0Weights[2][2][0] = -1;
		testFilter0Weights[2][2][1] = 1;
		testFilter0Weights[2][2][2] = 0;

		testFilter0 = new Filter(testFilter0Weights);
		testLayer.filters.add(testFilter0);//Add the filter to the list of filters in the layer

		testBias0 = 1;//Create a bias

		testLayer.biases.add(testBias0);//Add the bias to the list of biases in the layer

		//Create and initialize the second filter
		double[][][] testFilter1Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter1Weights[0][0][0] = -1;
		testFilter1Weights[0][0][1] = -1;
		testFilter1Weights[0][0][2] = -1;

		testFilter1Weights[0][1][0] = 1;
		testFilter1Weights[0][1][1] = 0;
		testFilter1Weights[0][1][2] = 1;

		testFilter1Weights[0][2][0] = 1;
		testFilter1Weights[0][2][1] = 1;
		testFilter1Weights[0][2][2] = 1;

		testFilter1Weights[1][0][0] = -1;
		testFilter1Weights[1][0][1] = -1;
		testFilter1Weights[1][0][2] = -1;

		testFilter1Weights[1][1][0] = -1;
		testFilter1Weights[1][1][1] = 1;
		testFilter1Weights[1][1][2] = 0;

		testFilter1Weights[1][2][0] = -1;
		testFilter1Weights[1][2][1] = 1;
		testFilter1Weights[1][2][2] = 1;

		testFilter1Weights[2][0][0] = 0;
		testFilter1Weights[2][0][1] = 0;
		testFilter1Weights[2][0][2] = 0;

		testFilter1Weights[2][1][0] = 0;
		testFilter1Weights[2][1][1] = -1;
		testFilter1Weights[2][1][2] = -1;

		testFilter1Weights[2][2][0] = 0;
		testFilter1Weights[2][2][1] = -1;
		testFilter1Weights[2][2][2] = -1;

		testFilter1 = new Filter(testFilter1Weights);
		testLayer.filters.add(testFilter1);//Add the filter to the list of filters in the layer

		testBias1 = 0;//Create a bias

		testLayer.biases.add(testBias1);//Add the bias to the list of biases in the layer

		//Create and initialize the third filter
		double[][][] testFilter2Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter2Weights[0][0][0] = 0;
		testFilter2Weights[0][0][1] = 1;
		testFilter2Weights[0][0][2] = 2;

		testFilter2Weights[0][1][0] = 1;
		testFilter2Weights[0][1][1] = 0;
		testFilter2Weights[0][1][2] = 1;

		testFilter2Weights[0][2][0] = 2;
		testFilter2Weights[0][2][1] = 1;
		testFilter2Weights[0][2][2] = 0;

		testFilter2Weights[1][0][0] = 1;
		testFilter2Weights[1][0][1] = -1;
		testFilter2Weights[1][0][2] = 0;

		testFilter2Weights[1][1][0] = -1;
		testFilter2Weights[1][1][1] = 1;
		testFilter2Weights[1][1][2] = -1;

		testFilter2Weights[1][2][0] = 0;
		testFilter2Weights[1][2][1] = -1;
		testFilter2Weights[1][2][2] = 1;

		testFilter2Weights[2][0][0] = -1;
		testFilter2Weights[2][0][1] = 1;
		testFilter2Weights[2][0][2] = 2;

		testFilter2Weights[2][1][0] = 1;
		testFilter2Weights[2][1][1] = 2;
		testFilter2Weights[2][1][2] = 1;

		testFilter2Weights[2][2][0] = 2;
		testFilter2Weights[2][2][1] = 1;
		testFilter2Weights[2][2][2] = -1;

		testFilter2 = new Filter(testFilter2Weights);
		testLayer.filters.add(testFilter2);//Add the filter to the list of filters in the layer

		testBias2 = -1;//Create a bias

		testLayer.biases.add(testBias2);//Add the bias to the list of biases in the layer

		//Create and initialize the array to use to store the output
		testOut = new double[testLayer.filters.size()];

	}

	@Test
	public void test() throws Exception {
		testLayer.full(testLayer.cells, testLayer.filters, testOut, 1, 0, testLayer.biases);

		//This is an array of values we expect to see in testOut
		double[] temp = new double[3];

		temp[0] = 0.0009110505;
		temp[1] = 0.9820138;
		temp[2] = 0.000000002061151;

		//Compare the temp array to the values stored in the output array
		for (int i = 0; i < temp.length; i++) {
			assertEquals(temp[i], testOut[i], 0);

		}
	}

}
