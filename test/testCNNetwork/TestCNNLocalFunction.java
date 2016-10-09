package testCNNetwork;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.Filter;
import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests the "local" function found in cnnetwork.Layer.java
 * 
 */
public class TestCNNLocalFunction {

	Layer testLayer;

	Filter testFilter0, testFilter1, testFilter2, testFilter3, testFilter4, testFilter5, testFilter6, testFilter7,
			testFilter8, testFilter9, testFilter10, testFilter11;

	Cell testBias0, testBias1, testBias2, testBias3, testBias4, testBias5, testBias6, testBias7, testBias8, testBias9,
			testBias10, testBias11;

	Cell[][][] testOut;

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

		//Create and initialize the first filter's weights
		double[][][] testFilter0Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter0Weights[0][0][0] = 1;
		testFilter0Weights[0][0][1] = 1;

		testFilter0Weights[0][1][0] = 0;
		testFilter0Weights[0][1][1] = 1;

		//Create and initialize the second filter's weights
		double[][][] testFilter1Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter1Weights[0][0][0] = -1;
		testFilter1Weights[0][0][1] = -1;

		testFilter1Weights[0][1][0] = 1;
		testFilter1Weights[0][1][1] = 0;

		//Create and initialize the third filter's weights
		double[][][] testFilter2Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter2Weights[0][0][0] = 0;
		testFilter2Weights[0][0][1] = 0;

		testFilter2Weights[0][1][0] = 1;
		testFilter2Weights[0][1][1] = 0;

		//Actually create the first three filters
		testFilter0 = new Filter(testFilter0Weights);
		testFilter1 = new Filter(testFilter1Weights);
		testFilter2 = new Filter(testFilter2Weights);
		
		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter0);
		testLayer.filters.add(testFilter1);
		testLayer.filters.add(testFilter2);
		
		//Create the biases for the first three filters
		testBias0 = new Cell(1);
		testBias1 = new Cell(0);
		testBias2 = new Cell(-1);

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias0);
		testLayer.biases.add(testBias1);
		testLayer.biases.add(testBias2);

		//Create and initialize the fourth filter's weights
		double[][][] testFilter3Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter3Weights[0][0][0] = 0;
		testFilter3Weights[0][0][1] = 0;

		testFilter3Weights[0][1][0] = 0;
		testFilter3Weights[0][1][1] = -1;

		//Create and initialize the fifth filter's weights
		double[][][] testFilter4Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter4Weights[0][0][0] = 1;
		testFilter4Weights[0][0][1] = -1;

		testFilter4Weights[0][1][0] = 0;
		testFilter4Weights[0][1][1] = -1;

		//Create and initialize the sixth filter's weights
		double[][][] testFilter5Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter5Weights[0][0][0] = 1;
		testFilter5Weights[0][0][1] = 1;

		testFilter5Weights[0][1][0] = -1;
		testFilter5Weights[0][1][1] = 0;

		//Actually create filters 4, 5 and 6
		testFilter3 = new Filter(testFilter3Weights);
		testFilter4 = new Filter(testFilter4Weights);
		testFilter5 = new Filter(testFilter5Weights);
				
		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter3);
		testLayer.filters.add(testFilter4);
		testLayer.filters.add(testFilter5);

		//Create the biases for the filters
		testBias3 = new Cell(2);
		testBias4 = new Cell(1);
		testBias5 = new Cell(0);

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias3);
		testLayer.biases.add(testBias4);
		testLayer.biases.add(testBias5);

		//Create and initialize the seventh filter's weights
		double[][][] testFilter6Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter6Weights[0][0][0] = 0;
		testFilter6Weights[0][0][1] = 0;

		testFilter6Weights[0][1][0] = -1;
		testFilter6Weights[0][1][1] = -1;

		//Create and initialize the eighth filter's weights
		double[][][] testFilter7Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter7Weights[0][0][0] = 1;
		testFilter7Weights[0][0][1] = 0;

		testFilter7Weights[0][1][0] = 0;
		testFilter7Weights[0][1][1] = 0;

		//Create and initialize the ninth filter's weights
		double[][][] testFilter8Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter8Weights[0][0][0] = -1;
		testFilter8Weights[0][0][1] = 1;

		testFilter8Weights[0][1][0] = -1;
		testFilter8Weights[0][1][1] = 1;

		//Actually create filters 7, 8 and 9
		testFilter6 = new Filter(testFilter6Weights);
		testFilter7 = new Filter(testFilter7Weights);
		testFilter8 = new Filter(testFilter8Weights);
		
		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter6);
		testLayer.filters.add(testFilter7);
		testLayer.filters.add(testFilter8);

		//Create the biases for the filters
		testBias6 = new Cell(0);
		testBias7 = new Cell(1);
		testBias8 = new Cell(-1);

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias6);
		testLayer.biases.add(testBias7);
		testLayer.biases.add(testBias8);

		//Create and initialize the tenth filter's weights
		double[][][] testFilter9Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter9Weights[0][0][0] = 1;
		testFilter9Weights[0][0][1] = 1;

		testFilter9Weights[0][1][0] = -1;
		testFilter9Weights[0][1][1] = 1;

		//Create and initialize the eleventh filter's weights
		double[][][] testFilter10Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter10Weights[0][0][0] = 0;
		testFilter10Weights[0][0][1] = 1;

		testFilter10Weights[0][1][0] = -1;
		testFilter10Weights[0][1][1] = 0;

		//Create and initialize the twelfth filter's weights
		double[][][] testFilter11Weights = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter11Weights[0][0][0] = 0;
		testFilter11Weights[0][0][1] = 0;

		testFilter11Weights[0][1][0] = 1;
		testFilter11Weights[0][1][1] = 0;
		
		//Actually create the last three filters (#s 10, 11, and 12)
		testFilter9 = new Filter(testFilter9Weights);
		testFilter10 = new Filter(testFilter10Weights);
		testFilter11 = new Filter(testFilter11Weights);

		//Add the filters to the list of filters in the layer
		testLayer.filters.add(testFilter9);
		testLayer.filters.add(testFilter10);
		testLayer.filters.add(testFilter11);

		//Create the biases for the filters
		testBias9 = new Cell(3);
		testBias10 = new Cell(1);
		testBias11 = new Cell(0);

		//Add the biases to the list of biases in the layer
		testLayer.biases.add(testBias9);
		testLayer.biases.add(testBias10);
		testLayer.biases.add(testBias11);

		//Calculate the width and depth needed for the output array
		int width = ((testLayer.cells[0][0].length - testLayer.Fcollumns + (2 * testLayer.pad)) / testLayer.step) + 1;
		int depth = testLayer.depth;

		//Create and initialize the array needed to hold the output
		testOut = new Cell[depth][width][width];
		
		// Initialize the cells because java won't do it for you
		// Depth
		for (int l = 0; l < testOut.length; l++) {
			// Row
			for (int m = 0; m < testOut[0].length; m++) {
				// Column
				for (int n = 0; n < testOut[0][0].length; n++) {
					testOut[l][m][n] = new Cell();
				}
			}
		}
		
	}

	@Test
	public void test() throws Exception {
		testLayer.local(testLayer.cells, testLayer.filters, testOut, 1, 0, testLayer.biases, true);

		//This is an array of values we expect to see in testOut
		double[][][] temp = new double[3][2][2];

		temp[0][0][0] = 0.9933072;
		temp[0][0][1] = 0.5;

		temp[0][1][0] = 0.2689414;
		temp[0][1][1] = 0.8807971;

		temp[1][0][0] = 0.5;
		temp[1][0][1] = 0.5;

		temp[1][1][0] = 0.04742587;
		temp[1][1][1] = 0.9525741;

		temp[2][0][0] = 0.5;
		temp[2][0][1] = 0.9820138;

		temp[2][1][0] = 0.9525741;
		temp[2][1][1] = 0.7310586;

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
