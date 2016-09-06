package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Filter;
import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests the "compute" function found in cnnetwork.Layer.java
 * 
 */
public class TestCNNComputeFunction {

	Layer testLayer;
	double[][][] testFilterWeights;
	
	double testBias;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(3, 3, 3, 2, 2, 3, 4, 1, 0, LayerType.CONV);

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

		//Create and initialize the filter weights
		testFilterWeights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		//Set the weights (entries in the filter)
		testFilterWeights[0][0][0] = 1;
		testFilterWeights[0][0][1] = 1;

		testFilterWeights[0][1][0] = 0;
		testFilterWeights[0][1][1] = 1;

		testFilterWeights[1][0][0] = 1;
		testFilterWeights[1][0][1] = -1;

		testFilterWeights[1][1][0] = 0;
		testFilterWeights[1][1][1] = -1;

		testFilterWeights[2][0][0] = -1;
		testFilterWeights[2][0][1] = 1;

		testFilterWeights[2][1][0] = -1;
		testFilterWeights[2][1][1] = 1;
		
		Filter testFilter = new Filter(testFilterWeights);//Create the filter with the newly made weights

		testLayer.filters.add(testFilter);//Actually add the filter to the list of filters in the layer

		testBias = 1;//Create a bias

		testLayer.biases.add(testBias);//Add the bias to the list of biases in the filter
	}

	@Test
	public void testCompute() throws Exception {
		double result0 = testLayer.compute(testLayer.filters.get(0), testLayer.cells, 0, 0, 0,
				testLayer.biases.get(0));
		assertEquals(result0, 0.006692849, 0);
	
		double result1 = testLayer.compute(testLayer.filters.get(0), testLayer.cells, 1, 1, 0,
				testLayer.biases.get(0));
		assertEquals(result1, 0.01798621, 0);

		double result2 = testLayer.compute(testLayer.filters.get(0), testLayer.cells, 0, 1, 0,
				testLayer.biases.get(0));
		assertEquals(result2, 0.01798621, 0);
	}

}
