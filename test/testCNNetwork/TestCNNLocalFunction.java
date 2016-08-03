package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;
import cnnetwork.NetworkCell;

/**
 * This tests the "local" function found in cnnetwork.Layer.java
 * 
 */
public class TestCNNLocalFunction {

	Layer testLayer;

	double[][][] testFilter0, testFilter1, testFilter2, testFilter3, testFilter4, testFilter5, testFilter6, testFilter7,
			testFilter8, testFilter9, testFilter10, testFilter11;

	double testBias0, testBias1, testBias2, testBias3, testBias4, testBias5, testBias6, testBias7, testBias8, testBias9,
			testBias10, testBias11;

	NetworkCell[][][] testOut;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(3, 3, 3, 2, 2, 1, 12, 1, 0, LayerType.LOCAL);

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

		//Create and initialize the first filter
		testFilter0 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter0[0][0][0] = 1;
		testFilter0[0][0][1] = 1;

		testFilter0[0][1][0] = 0;
		testFilter0[0][1][1] = 1;

		//Create and initialize the second filter
		testFilter1 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter1[0][0][0] = -1;
		testFilter1[0][0][1] = -1;

		testFilter1[0][1][0] = 1;
		testFilter1[0][1][1] = 0;

		//Create and initialize the third filter
		testFilter2 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter2[0][0][0] = 0;
		testFilter2[0][0][1] = 0;

		testFilter2[0][1][0] = 1;
		testFilter2[0][1][1] = 0;

		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter0);
		testLayer.filters.add(testFilter1);
		testLayer.filters.add(testFilter2);

		//Create the biases for the first three filters
		testBias0 = 1;
		testBias1 = 0;
		testBias2 = -1;

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias0);
		testLayer.biases.add(testBias1);
		testLayer.biases.add(testBias2);

		//Create and initialize the fourth filter
		testFilter3 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter3[0][0][0] = 0;
		testFilter3[0][0][1] = 0;

		testFilter3[0][1][0] = 0;
		testFilter3[0][1][1] = -1;

		//Create and initialize the fifth filter
		testFilter4 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter4[0][0][0] = 1;
		testFilter4[0][0][1] = -1;

		testFilter4[0][1][0] = 0;
		testFilter4[0][1][1] = -1;

		//Create and initialize the sixth filter
		testFilter5 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter5[0][0][0] = 1;
		testFilter5[0][0][1] = 1;

		testFilter5[0][1][0] = -1;
		testFilter5[0][1][1] = 0;

		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter3);
		testLayer.filters.add(testFilter4);
		testLayer.filters.add(testFilter5);

		//Create the biases for the filters
		testBias3 = 2;
		testBias4 = 1;
		testBias5 = 0;

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias3);
		testLayer.biases.add(testBias4);
		testLayer.biases.add(testBias5);

		//Create and initialize the seventh filter
		testFilter6 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter6[0][0][0] = 0;
		testFilter6[0][0][1] = 0;

		testFilter6[0][1][0] = -1;
		testFilter6[0][1][1] = -1;

		//Create and initialize the eighth filter
		testFilter7 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter7[0][0][0] = 1;
		testFilter7[0][0][1] = 0;

		testFilter7[0][1][0] = 0;
		testFilter7[0][1][1] = 0;

		//Create and initialize the ninth filter
		testFilter8 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter8[0][0][0] = -1;
		testFilter8[0][0][1] = 1;

		testFilter8[0][1][0] = -1;
		testFilter8[0][1][1] = 1;

		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter6);
		testLayer.filters.add(testFilter7);
		testLayer.filters.add(testFilter8);

		//Create the biases for the filters
		testBias6 = 0;
		testBias7 = 1;
		testBias8 = -1;

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias6);
		testLayer.biases.add(testBias7);
		testLayer.biases.add(testBias8);

		//Create and initialize the tenth filter
		testFilter9 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter9[0][0][0] = 1;
		testFilter9[0][0][1] = 1;

		testFilter9[0][1][0] = -1;
		testFilter9[0][1][1] = 1;

		//Create and initialize the eleventh filter
		testFilter10 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter10[0][0][0] = 0;
		testFilter10[0][0][1] = 1;

		testFilter10[0][1][0] = -1;
		testFilter10[0][1][1] = 0;

		//Create and initialize the twelfth filter
		testFilter11 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter11[0][0][0] = 0;
		testFilter11[0][0][1] = 0;

		testFilter11[0][1][0] = 1;
		testFilter11[0][1][1] = 0;

		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter9);
		testLayer.filters.add(testFilter10);
		testLayer.filters.add(testFilter11);

		//Create the biases for the filters
		testBias9 = 3;
		testBias10 = 1;
		testBias11 = 0;

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias9);
		testLayer.biases.add(testBias10);
		testLayer.biases.add(testBias11);

		//Calculate the width and depth needed for the output array
		int width = ((testLayer.cells[0][0].length - testLayer.Fcollumns + (2 * testLayer.pad)) / testLayer.step) + 1;
		int depth = testLayer.depth;

		//Create and initialize the array needed to hold the output
		testOut = new NetworkCell[depth][width][width];
		
		//Don't forget to initialize the cells- java won't do it for you.
		// Depth
		for (int d = 0; d < depth; d++) {
			// Row
			for (int r = 0; r < width; r ++) {
				// Column
				for (int c = 0; c  <width; c ++) {
					testOut[d][r][c] = new NetworkCell();

				}

			}

		}
	}

	@Test
	public void test() {
		testLayer.local(testLayer.cells, testLayer.filters, testOut, 1, 0, testLayer.biases);

		//This is an array of values we expect to see in testOut
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

		//Compare the temp array to the values stored in the output array
		for (int i = 0; i < temp.length; i++) {
			for (int j = 0; j < temp[0].length; j++) {
				for (int k = 0; k < temp[0][0].length; k++) {
					assertEquals(temp[i][j][k], testOut[i][j][k].value, 0);
				}
			}
		}
	}
}
