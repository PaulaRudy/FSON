package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNComputeMaxFunction {

	Layer testLayer;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		testLayer = new Layer(3, 3, 3, 2, 2, 3, 1, 2, 0, LayerType.MAXPOOL);

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
		testLayer.values[2][1][1] = 1;
		testLayer.values[2][1][2] = 1;

		testLayer.values[2][2][0] = 0;
		testLayer.values[2][2][1] = 1;
		testLayer.values[2][2][2] = 1;

	}

	@Test
	public void testComputeMax() {
		double result0 = testLayer.computeMax(testLayer.values, 0, 0, 0, 2);
		assertEquals(result0, 2, 0);

		double result1 = testLayer.computeMax(testLayer.values, 1, 0, 0, 2);
		assertEquals(result1, 2, 0);

		double result2 = testLayer.computeMax(testLayer.values, 0, 0, 2, 2);
		assertEquals(result2, 2, 0);

		double result3 = testLayer.computeMax(testLayer.values, 1, 1, 2, 2);
		assertEquals(result3, 1, 0);

	}

}
