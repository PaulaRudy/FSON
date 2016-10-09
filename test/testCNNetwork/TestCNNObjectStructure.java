/**
 * 
 */
package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.Filter;
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

		//Create and initialize the filters' weights
		double[][][] testFilter1Weights = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter1Weights[1][0][0] = 1.0;

		double[][][] testFilter2Weights = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter2Weights[0][1][0] = 1.0;

		double[][][] testFilter3Weights = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter3Weights[0][0][1] = 1.0;

		double[][][] testFilter4Weights = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter4Weights[2][2][2] = 1.0;
		
		//Actually create the filters
		Filter testFilter1 = new Filter(testFilter1Weights);
		Filter testFilter2 = new Filter(testFilter2Weights);
		Filter testFilter3 = new Filter(testFilter3Weights);
		Filter testFilter4 = new Filter(testFilter4Weights);

		//Add the filters to the layer
		test.filters.add(testFilter1);
		test.filters.add(testFilter2);
		test.filters.add(testFilter3);
		test.filters.add(testFilter4);

		//Test that the filters are still there and in the right place 
		assertEquals(test.filters.get(0).weights[0][0][0], 0, 0);
		assertEquals(test.filters.get(0).weights[1][0][0], 1, 0);
		assertEquals(test.filters.get(1).weights[0][0][0], 0, 0);
		assertEquals(test.filters.get(1).weights[0][1][0], 1, 0);
		assertEquals(test.filters.get(2).weights[0][0][0], 0, 0);
		assertEquals(test.filters.get(2).weights[0][0][1], 1, 0);
		assertEquals(test.filters.get(3).weights[0][0][0], 0, 0);
		assertEquals(test.filters.get(3).weights[2][2][2], 1, 0);
		assertEquals(test.filters.get(0).weights[1][0][0], 1, 0);
	}

	/**
	 * Test adding biases
	 */
	@Test
	public void testBiases() {
		//Create and initialize the biases
		Cell testBias = new Cell(0.45);
		Cell testBias2 = new Cell(0.33);
		
		//Add the biases to the layer
		test.biases.add(testBias);
		test.biases.add(testBias2);
		
		//Test that the biases are still there and in the right place 
		assertEquals(test.biases.get(0).value, 0.45, 0);
		assertEquals(test.biases.get(1).value, 0.33, 0);
	}

}
