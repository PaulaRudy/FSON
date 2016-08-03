package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;


/**
 * This tests the "computeMax" function found in cnnetwork.Layer.java
 *
 */
public class TestCNNComputeMaxFunction {

	Layer testLayer;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(3, 3, 3, 2, 2, 3, 1, 2, 0, LayerType.MAXPOOL);

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
		testLayer.cells[2][1][1].value = 1;
		testLayer.cells[2][1][2].value = 1;

		testLayer.cells[2][2][0].value = 0;
		testLayer.cells[2][2][1].value = 1;
		testLayer.cells[2][2][2].value = 1;

	}

	@Test
	public void testComputeMax() {
		double result0 = testLayer.computeMax(testLayer.cells, 0, 0, 0, 2);
		assertEquals(result0, 2, 0);

		double result1 = testLayer.computeMax(testLayer.cells, 1, 0, 0, 2);
		assertEquals(result1, 2, 0);

		double result2 = testLayer.computeMax(testLayer.cells, 0, 0, 2, 2);
		assertEquals(result2, 2, 0);

		double result3 = testLayer.computeMax(testLayer.cells, 1, 1, 2, 2);
		assertEquals(result3, 1, 0);

	}

}
