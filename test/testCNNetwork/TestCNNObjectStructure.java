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
 * @author 
 *
 */
public class TestCNNObjectStructure {

	Layer test;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		test = new Layer(3, 3, 3, 3, 3, 3, 1, 0, 0, LayerType.FULLY);
	}

	@Test
	public void testFilters() {
		double[][][] testFilter = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter[1][0][0] = 1.0;

		double[][][] testFilter2 = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter2[0][1][0] = 1.0;

		double[][][] testFilter3 = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter3[0][0][1] = 1.0;

		double[][][] testFilter4 = new double[test.Fdepth][test.Frows][test.Fcollumns];

		testFilter4[2][2][2] = 1.0;

		test.filters.add(testFilter);
		test.filters.add(testFilter2);
		test.filters.add(testFilter3);
		test.filters.add(testFilter4);
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

	@Test
	public void testBiases() {
		Double testBias = 0.45;
		Double testBias2 = 0.33;
		test.biases.add(testBias);
		test.biases.add(testBias2);
		assertEquals(test.biases.get(0), 0.45, 0);
		assertEquals(test.biases.get(1), 0.33, 0);
	}

}
