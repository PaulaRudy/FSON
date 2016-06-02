package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNComputeFunction {

	Layer testLayer;
	double[][][] testFilter;
	
	double testBias;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		testLayer = new Layer(3, 3, 3, 2, 2, 3, 4, 1, 0, LayerType.CONV);

		// Layer that acts as input

		testLayer.values[0][0][0] = 2;
		testLayer.values[0][0][1] = 0;
		testLayer.values[0][0][2] = 2;

		testLayer.values[0][1][0] = 2;
		testLayer.values[0][1][1] = 2;
		testLayer.values[0][1][2] = 2;

		testLayer.values[0][2][0] = 0;
		testLayer.values[0][2][1] = 1;
		testLayer.values[0][2][2] = 0;

		testLayer.values[1][0][0] = 1;
		testLayer.values[1][0][1] = 0;
		testLayer.values[1][0][2] = 2;

		testLayer.values[1][1][0] = 0;
		testLayer.values[1][1][1] = 2;
		testLayer.values[1][1][2] = 0;

		testLayer.values[1][2][0] = 2;
		testLayer.values[1][2][1] = 1;
		testLayer.values[1][2][2] = 2;

		testLayer.values[2][0][0] = 1;
		testLayer.values[2][0][1] = 2;
		testLayer.values[2][0][2] = 0;

		testLayer.values[2][1][0] = 2;
		testLayer.values[2][1][1] = 2;
		testLayer.values[2][1][2] = 1;

		testLayer.values[2][2][0] = 0;
		testLayer.values[2][2][1] = 1;
		testLayer.values[2][2][2] = 1;

		testFilter = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

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

		testLayer.filters.add(testFilter);

		testBias = 1;

		testLayer.biases.add(testBias);
	}

	@Test
	public void testCompute() {
		double result0 = testLayer.compute(testLayer.filters.get(0), testLayer.values, 0, 0, 0,
				testLayer.biases.get(0));
		assertEquals(result0, 7, 0);

		double result1 = testLayer.compute(testLayer.filters.get(0), testLayer.values, 1, 1, 0,
				testLayer.biases.get(0));
		assertEquals(result1, 6, 0);

		double result2 = testLayer.compute(testLayer.filters.get(0), testLayer.values, 0, 1, 0,
				testLayer.biases.get(0));
		assertEquals(result2, 6, 0);
	}

}
