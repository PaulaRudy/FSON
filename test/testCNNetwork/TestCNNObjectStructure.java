/**
 * 
 */
package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests adding the filters and biases to a layer.
 *
 */
public class TestCNNObjectStructure {

	Layer test;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		//Create and initialize the layer to use for testing
		test = new Layer(3, 3, 3, 3, 3, 3, 1, 0, 0, LayerType.FULLY);
	}

	/**
	 * Test adding filters
	 */
	@Test
	public void testFilters() {

		//Create and initialize the filters
		double[][][] testFilter = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter[1][0][0] = 1.0;

		double[][][] testFilter2 = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter2[0][1][0] = 1.0;

		double[][][] testFilter3 = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter3[0][0][1] = 1.0;

		double[][][] testFilter4 = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter4[2][2][2] = 1.0;

		//Add the filters to the layer
		test.filters.add(testFilter);
		test.filters.add(testFilter2);
		test.filters.add(testFilter3);
		test.filters.add(testFilter4);

		//Test that the filters are still there and in the right place 
		assertEquals(test.filters.get(0)[0][0][0], 0, 0);
		assertEquals(test.filters.get(0)[1][0][0], 1, 0);
		assertEquals(test.filters.get(1)[0][0][0], 0, 0);
		assertEquals(test.filters.get(1)[0][1][0], 1, 0);
		assertEquals(test.filters.get(2)[0][0][0], 0, 0);
		assertEquals(test.filters.get(2)[0][0][1], 1, 0);
		assertEquals(test.filters.get(3)[0][0][0], 0, 0);
		assertEquals(test.filters.get(3)[2][2][2], 1, 0);
		assertEquals(test.filters.get(0)[1][0][0], 1, 0);
	}

	/**
	 * Test adding biases
	 */
	@Test
	public void testBiases() {
		//Create and initialize the biases
		Double testBias = 0.45;
		Double testBias2 = 0.33;
		
		//Add the biases to the layer
		test.biases.add(testBias);
		test.biases.add(testBias2);
		
		//Test that the biases are still there and in the right place 
		assertEquals(test.biases.get(0), 0.45, 0);
		assertEquals(test.biases.get(1), 0.33, 0);
	}

}
