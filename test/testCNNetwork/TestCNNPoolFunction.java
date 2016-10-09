package testCNNetwork;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.Filter;
import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests the "pool" function in cnnetwork.Layer.java
 *
 */
public class TestCNNPoolFunction {

	
	Layer testLayer;
	
	Cell[][][] testOut;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(4, 4, 3, 2, 2, 3, 12, 2, 0, LayerType.MAXPOOL);
		
		//Set the values of all the cells
		testLayer.cells[0][0][0].value = 0;
		testLayer.cells[0][0][1].value = 0;
		testLayer.cells[0][0][2].value = 2;
		testLayer.cells[0][0][3].value = 1;

		testLayer.cells[0][1][0].value = 0;
		testLayer.cells[0][1][1].value = 1;
		testLayer.cells[0][1][2].value = 0;
		testLayer.cells[0][1][3].value = 1;

		testLayer.cells[0][2][0].value = 1;
		testLayer.cells[0][2][1].value = 4;
		testLayer.cells[0][2][2].value = 0;
		testLayer.cells[0][2][3].value = 0;
		
		testLayer.cells[0][3][0].value = 1;
		testLayer.cells[0][3][1].value = 1;
		testLayer.cells[0][3][2].value = 0;
		testLayer.cells[0][3][3].value = 0;
		
		
		testLayer.cells[1][0][0].value = 3;
		testLayer.cells[1][0][1].value = 2;
		testLayer.cells[1][0][2].value = 1;
		testLayer.cells[1][0][3].value = 0;

		testLayer.cells[1][1][0].value = 1;
		testLayer.cells[1][1][1].value = 0;
		testLayer.cells[1][1][2].value = 2;
		testLayer.cells[1][1][3].value = 0;

		testLayer.cells[1][2][0].value = 2;
		testLayer.cells[1][2][1].value = 0;
		testLayer.cells[1][2][2].value = 0;
		testLayer.cells[1][2][3].value = 0;
		
		testLayer.cells[1][3][0].value = 5;
		testLayer.cells[1][3][1].value = 3;
		testLayer.cells[1][3][2].value = 0;
		testLayer.cells[1][3][3].value = 3;
		
		
		testLayer.cells[2][0][0].value = 0;
		testLayer.cells[2][0][1].value = 0;
		testLayer.cells[2][0][2].value = 0;
		testLayer.cells[2][0][3].value = 0;

		testLayer.cells[2][1][0].value = 0;
		testLayer.cells[2][1][1].value = 0;
		testLayer.cells[2][1][2].value = 0;
		testLayer.cells[2][1][3].value = 0;

		testLayer.cells[2][2][0].value = 0;
		testLayer.cells[2][2][1].value = 0;
		testLayer.cells[2][2][2].value = 0;
		testLayer.cells[2][2][3].value = 0;
		
		testLayer.cells[2][3][0].value = 0;
		testLayer.cells[2][3][1].value = 0;
		testLayer.cells[2][3][2].value = 1;
		testLayer.cells[2][3][3].value = 0;
		

		//Calculate the width of the array to use to store the output
		int width = ((testLayer.cells[0][0].length -testLayer.Fcollumns + (2* testLayer.pad))/testLayer.step)+ 1;
		
		//Create and initialize the array to use to store the output
		testOut = new Cell[testLayer.cells.length][width][width];
		
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
		
		// Create and initialize the "filters"
		// Because this is a maxpool layer, these filters are only used to
		// record connections for use during backpropagation, and we don't need
		// any biases.
		for (int i = 0; i < testLayer.K; i++) {
			// Create the filter weights
			double[][][] newFilterWeights = new double[1][testLayer.Frows][testLayer.Fcollumns];

			// (java will initialize them to 0)

			Filter newFilter = new Filter(newFilterWeights);// Use the default constructor with the newly created filter weights
			testLayer.filters.add(newFilter);// Actually add the filter to the list of filters in the layer
		}
		
		
	}

	@Test
	public void test() {
		testLayer.pool(testLayer.cells, testLayer.filters, testOut, testLayer.step, testLayer.Fcollumns, true);
		
		//This is an array of values we expect to see in testOut
		double[][][] temp = new double[3][2][2];
		
		temp[0][0][0] = 1;
		temp[0][0][1] = 2;

		temp[0][1][0] = 4;
		temp[0][1][1] = 0;

		temp[1][0][0] = 3;
		temp[1][0][1] = 2;

		temp[1][1][0] = 5;
		temp[1][1][1] = 3;

		temp[2][0][0] = 0;
		temp[2][0][1] = 0;

		temp[2][1][0] = 0;
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
