package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNConvolutionFunction {

	Layer testLayer;

	double[][][] testFilter0, testFilter1, testFilter2, testFilter3;

	double testBias0, testBias1, testBias2, testBias3;

	double[][][] testOut;

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

		testFilter0 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter0[0][0][0] = 1;
		testFilter0[0][0][1] = 1;

		testFilter0[0][1][0] = 0;
		testFilter0[0][1][1] = 1;

		testFilter0[1][0][0] = 1;
		testFilter0[1][0][1] = -1;

		testFilter0[1][1][0] = 0;
		testFilter0[1][1][1] = -1;

		testFilter0[2][0][0] = -1;
		testFilter0[2][0][1] = 1;

		testFilter0[2][1][0] = -1;
		testFilter0[2][1][1] = 1;

		testLayer.filters.add(testFilter0);

		testBias0 = 1;

		testLayer.biases.add(testBias0);

		testFilter1 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter1[0][0][0] = -1;
		testFilter1[0][0][1] = -1;

		testFilter1[0][1][0] = 1;
		testFilter1[0][1][1] = 0;

		testFilter1[1][0][0] = 1;
		testFilter1[1][0][1] = 1;

		testFilter1[1][1][0] = -1;
		testFilter1[1][1][1] = 0;

		testFilter1[2][0][0] = 1;
		testFilter1[2][0][1] = 1;

		testFilter1[2][1][0] = -1;
		testFilter1[2][1][1] = 1;

		testLayer.filters.add(testFilter1);

		testBias1 = 0;

		testLayer.biases.add(testBias1);

		testFilter2 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter2[0][0][0] = 0;
		testFilter2[0][0][1] = 0;

		testFilter2[0][1][0] = 1;
		testFilter2[0][1][1] = 0;

		testFilter2[1][0][0] = 0;
		testFilter2[1][0][1] = 0;

		testFilter2[1][1][0] = -1;
		testFilter2[1][1][1] = -1;

		testFilter2[2][0][0] = 0;
		testFilter2[2][0][1] = 1;

		testFilter2[2][1][0] = -1;
		testFilter2[2][1][1] = 0;

		testLayer.filters.add(testFilter2);

		testBias2 = -1;

		testLayer.biases.add(testBias2);

		testFilter3 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter3[0][0][0] = 0;
		testFilter3[0][0][1] = 0;

		testFilter3[0][1][0] = 0;
		testFilter3[0][1][1] = -1;

		testFilter3[1][0][0] = 1;
		testFilter3[1][0][1] = 0;

		testFilter3[1][1][0] = 0;
		testFilter3[1][1][1] = 0;

		testFilter3[2][0][0] = 0;
		testFilter3[2][0][1] = 0;

		testFilter3[2][1][0] = 1;
		testFilter3[2][1][1] = 0;

		testLayer.filters.add(testFilter3);

		testBias3 = -2;

		testLayer.biases.add(testBias3);

		int width = ((testLayer.values[0][0].length - testLayer.Fcollumns + (2 * testLayer.pad)) / testLayer.step) + 1;

		testOut = new double[testLayer.K][width][width];
	}

	@Test
	public void test() {
		testLayer.convolution(testLayer.values, testLayer.filters, testOut, 1, 0, testLayer.biases);

		double[][][] temp = new double[4][2][2];

		temp[0][0][0] = 5;
		temp[0][0][1] = 0;

		temp[0][1][0] = 4;
		temp[0][1][1] = 4;

		temp[1][0][0] = 4;
		temp[1][0][1] = 1;

		temp[1][1][0] = 1;
		temp[1][1][1] = 1;

		temp[2][0][0] = -1;
		temp[2][0][1] = -3;

		temp[2][1][0] = -2;
		temp[2][1][1] = -3;

		temp[3][0][0] = -1;
		temp[3][0][1] = -2;

		temp[3][1][0] = -3;
		temp[3][1][1] = 1;

		for (int i = 0; i < temp.length; i++) {
			for (int j = 0; j < temp[0].length; j++) {
				for (int k = 0; k < temp[0][0].length; k++) {
					assertEquals(temp[i][j][k], testOut[i][j][k], 0);
				}
			}
		}
	}

}