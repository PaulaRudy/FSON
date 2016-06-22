package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNLocalFunction {

	Layer testLayer;

	double[][][] testFilter0, testFilter1, testFilter2, testFilter3, testFilter4, testFilter5, testFilter6, testFilter7,
			testFilter8, testFilter9, testFilter10, testFilter11;

	double testBias0, testBias1, testBias2, testBias3, testBias4, testBias5, testBias6, testBias7, testBias8, testBias9,
			testBias10, testBias11;

	double[][][] testOut;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		testLayer = new Layer(3, 3, 3, 2, 2, 1, 12, 1, 0, LayerType.LOCAL);

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

		testFilter1 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter1[0][0][0] = -1;
		testFilter1[0][0][1] = -1;

		testFilter1[0][1][0] = 1;
		testFilter1[0][1][1] = 0;

		testFilter2 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter2[0][0][0] = 0;
		testFilter2[0][0][1] = 0;

		testFilter2[0][1][0] = 1;
		testFilter2[0][1][1] = 0;

		testLayer.filters.add(testFilter0);
		testLayer.filters.add(testFilter1);
		testLayer.filters.add(testFilter2);

		testBias0 = 1;
		testBias1 = 0;
		testBias2 = -1;

		testLayer.biases.add(testBias0);
		testLayer.biases.add(testBias1);
		testLayer.biases.add(testBias2);

		testFilter3 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter3[0][0][0] = 0;
		testFilter3[0][0][1] = 0;

		testFilter3[0][1][0] = 0;
		testFilter3[0][1][1] = -1;

		testFilter4 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter4[0][0][0] = 1;
		testFilter4[0][0][1] = -1;

		testFilter4[0][1][0] = 0;
		testFilter4[0][1][1] = -1;

		testFilter5 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter5[0][0][0] = 1;
		testFilter5[0][0][1] = 1;

		testFilter5[0][1][0] = -1;
		testFilter5[0][1][1] = 0;

		testLayer.filters.add(testFilter3);
		testLayer.filters.add(testFilter4);
		testLayer.filters.add(testFilter5);

		testBias3 = 2;
		testBias4 = 1;
		testBias5 = 0;

		testLayer.biases.add(testBias3);
		testLayer.biases.add(testBias4);
		testLayer.biases.add(testBias5);

		testFilter6 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter6[0][0][0] = 0;
		testFilter6[0][0][1] = 0;

		testFilter6[0][1][0] = -1;
		testFilter6[0][1][1] = -1;

		testFilter7 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter7[0][0][0] = 1;
		testFilter7[0][0][1] = 0;

		testFilter7[0][1][0] = 0;
		testFilter7[0][1][1] = 0;

		testFilter8 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter8[0][0][0] = -1;
		testFilter8[0][0][1] = 1;

		testFilter8[0][1][0] = -1;
		testFilter8[0][1][1] = 1;

		testLayer.filters.add(testFilter6);
		testLayer.filters.add(testFilter7);
		testLayer.filters.add(testFilter8);

		testBias6 = 0;
		testBias7 = 1;
		testBias8 = -1;

		testLayer.biases.add(testBias6);
		testLayer.biases.add(testBias7);
		testLayer.biases.add(testBias8);

		testFilter9 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter9[0][0][0] = 1;
		testFilter9[0][0][1] = 1;

		testFilter9[0][1][0] = -1;
		testFilter9[0][1][1] = 1;

		testFilter10 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter10[0][0][0] = 0;
		testFilter10[0][0][1] = 1;

		testFilter10[0][1][0] = -1;
		testFilter10[0][1][1] = 0;

		testFilter11 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter11[0][0][0] = 0;
		testFilter11[0][0][1] = 0;

		testFilter11[0][1][0] = 1;
		testFilter11[0][1][1] = 0;

		testLayer.filters.add(testFilter9);
		testLayer.filters.add(testFilter10);
		testLayer.filters.add(testFilter11);

		testBias9 = 3;
		testBias10 = 1;
		testBias11 = 0;

		testLayer.biases.add(testBias9);
		testLayer.biases.add(testBias10);
		testLayer.biases.add(testBias11);

		int width = ((testLayer.values[0][0].length - testLayer.Fcollumns + (2 * testLayer.pad)) / testLayer.step) + 1;
		int depth = testLayer.depth;

		testOut = new double[depth][width][width];
	}

	@Test
	public void test() {
		testLayer.local(testLayer.values, testLayer.filters, testOut, 1, 0, testLayer.biases);

		double[][][] temp = new double[3][2][2];

		temp[0][0][0] = 5;
		temp[0][0][1] = 0;

		temp[0][1][0] = -1;
		temp[0][1][1] = 2;

		temp[1][0][0] = 0;
		temp[1][0][1] = 0;

		temp[1][1][0] = -3;
		temp[1][1][1] = 3;

		temp[2][0][0] = 0;
		temp[2][0][1] = 4;

		temp[2][1][0] = 3;
		temp[2][1][1] = 1;

		for (int i = 0; i < temp.length; i++) {
			for (int j = 0; j < temp[0].length; j++) {
				for (int k = 0; k < temp[0][0].length; k++) {
					assertEquals(temp[i][j][k], testOut[i][j][k], 0);
				}
			}
		}
	}
}
