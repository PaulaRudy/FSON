package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests the "compute" function found in cnnetwork.Layer.java
 * 
 */
public class TestCNNComputeFunction {

	Layer testLayer;
	double[][][] testFilter;
	
	double testBias;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(3, 3, 3, 2, 2, 3, 4, 1, 0, LayerType.CONV);

		//Set the values of all the cells

		testLayer.cells[0][0][0].value = 2;
		testLayer.cells[0][0][1].value = 0;
		testLayer.cells[0][0][2].value = 2;

		testLayer.cells[0][1][0].value = 2;
		testLayer.cells[0][1][1].value = 2;
		testLayer.cells[0][1][2].value = 2;

		testLayer.cells[0][2][0].value = 0;
		testLayer.cells[0][2][1].value = 1;
		testLayer.cells[0][2][2].value = 0;

		testLayer.cells[1][0][0].value = 1;
		testLayer.cells[1][0][1].value = 0;
		testLayer.cells[1][0][2].value = 2;

		testLayer.cells[1][1][0].value = 0;
		testLayer.cells[1][1][1].value = 2;
		testLayer.cells[1][1][2].value = 0;

		testLayer.cells[1][2][0].value = 2;
		testLayer.cells[1][2][1].value = 1;
		testLayer.cells[1][2][2].value = 2;

		testLayer.cells[2][0][0].value = 1;
		testLayer.cells[2][0][1].value = 2;
		testLayer.cells[2][0][2].value = 0;

		testLayer.cells[2][1][0].value = 2;
		testLayer.cells[2][1][1].value = 2;
		testLayer.cells[2][1][2].value = 1;

		testLayer.cells[2][2][0].value = 0;
		testLayer.cells[2][2][1].value = 1;
		testLayer.cells[2][2][2].value = 1;

		//Create and initialize the filter
		testFilter = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		//Set the weights (entries in the filter)
		testFilter[0][0][0] = 1;
		testFilter[0][0][1] = 1;

		testFilter[0][1][0] = 0;
		testFilter[0][1][1] = 1;

		testFilter[1][0][0] = 1;
		testFilter[1][0][1] = -1;

		testFilter[1][1][0] = 0;
		testFilter[1][1][1] = -1;

		testFilter[2][0][0] = -1;
		testFilter[2][0][1] = 1;

		testFilter[2][1][0] = -1;
		testFilter[2][1][1] = 1;

		testLayer.filters.add(testFilter);//Actually add the filter to the list of filters in the layer

		testBias = 1;//Create a bias

		testLayer.biases.add(testBias);//Add the bias to the list of biases in the filter
	}

	@Test
	public void testCompute() {
		double result0 = testLayer.compute(testLayer.filters.get(0), testLayer.cells, 0, 0, 0,
				testLayer.biases.get(0));
		assertEquals(result0, 5, 0);
	
		double result1 = testLayer.compute(testLayer.filters.get(0), testLayer.cells, 1, 1, 0,
				testLayer.biases.get(0));
		assertEquals(result1, 4, 0);

		double result2 = testLayer.compute(testLayer.filters.get(0), testLayer.cells, 0, 1, 0,
				testLayer.biases.get(0));
		assertEquals(result2, 4, 0);
	}

}
