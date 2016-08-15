package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests the "pool" function in cnnetwork.Layer.java
 *
 */
public class TestCNNPoolFunction {

	Layer testLayer;
	
	double[][][] testOut;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(4, 4, 3, 2, 2, 3, 4, 2, 0, LayerType.MAXPOOL);
		
		//Set the values of all the cells
		testLayer.cells[0][0][0] = 0;
		testLayer.cells[0][0][1] = 0;
		testLayer.cells[0][0][2] = 2;
		testLayer.cells[0][0][3] = 1;

		testLayer.cells[0][1][0] = 0;
		testLayer.cells[0][1][1] = 1;
		testLayer.cells[0][1][2] = 0;
		testLayer.cells[0][1][3] = 1;

		testLayer.cells[0][2][0] = 1;
		testLayer.cells[0][2][1] = 4;
		testLayer.cells[0][2][2] = 0;
		testLayer.cells[0][2][3] = 0;
		
		testLayer.cells[0][3][0] = 1;
		testLayer.cells[0][3][1] = 1;
		testLayer.cells[0][3][2] = 0;
		testLayer.cells[0][3][3] = 0;
		
		
		testLayer.cells[1][0][0] = 3;
		testLayer.cells[1][0][1] = 2;
		testLayer.cells[1][0][2] = 1;
		testLayer.cells[1][0][3] = 0;

		testLayer.cells[1][1][0] = 1;
		testLayer.cells[1][1][1] = 0;
		testLayer.cells[1][1][2] = 2;
		testLayer.cells[1][1][3] = 0;

		testLayer.cells[1][2][0] = 2;
		testLayer.cells[1][2][1] = 0;
		testLayer.cells[1][2][2] = 0;
		testLayer.cells[1][2][3] = 0;
		
		testLayer.cells[1][3][0] = 5;
		testLayer.cells[1][3][1] = 3;
		testLayer.cells[1][3][2] = 0;
		testLayer.cells[1][3][3] = 3;
		
		
		testLayer.cells[2][0][0] = 0;
		testLayer.cells[2][0][1] = 0;
		testLayer.cells[2][0][2] = 0;
		testLayer.cells[2][0][3] = 0;

		testLayer.cells[2][1][0] = 0;
		testLayer.cells[2][1][1] = 0;
		testLayer.cells[2][1][2] = 0;
		testLayer.cells[2][1][3] = 0;

		testLayer.cells[2][2][0] = 0;
		testLayer.cells[2][2][1] = 0;
		testLayer.cells[2][2][2] = 0;
		testLayer.cells[2][2][3] = 0;
		
		testLayer.cells[2][3][0] = 0;
		testLayer.cells[2][3][1] = 0;
		testLayer.cells[2][3][2] = 1;
		testLayer.cells[2][3][3] = 0;
		

		//Calculate the width of the array to use to store the output
		int width = ((testLayer.cells[0][0].length -testLayer.Fcollumns + (2* testLayer.pad))/testLayer.step)+ 1;
		
		//Create and initialize the array to use to store the output
		testOut = new double[testLayer.cells.length][width][width];
		
	}

	@Test
	public void test() {
		testLayer.pool(testLayer.cells, testOut, testLayer.step, testLayer.Fcollumns);
		
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
					assertEquals(temp[i][j][k], testOut[i][j][k], 0); 
				}
			}
		}
	}

}
