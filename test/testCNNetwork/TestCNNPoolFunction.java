package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNPoolFunction {

	Layer testLayer;
	
	double[][][] testOut;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		testLayer = new Layer(4, 4, 3, 2, 2, 3, 4, 2, 0, LayerType.MAXPOOL);
		

		// Layer that acts as input

		testLayer.values[0][0][0] = 0;
		testLayer.values[0][0][1] = 0;
		testLayer.values[0][0][2] = 2;
		testLayer.values[0][0][3] = 1;

		testLayer.values[0][1][0] = 0;
		testLayer.values[0][1][1] = 1;
		testLayer.values[0][1][2] = 0;
		testLayer.values[0][1][3] = 1;

		testLayer.values[0][2][0] = 1;
		testLayer.values[0][2][1] = 4;
		testLayer.values[0][2][2] = 0;
		testLayer.values[0][2][3] = 0;
		
		testLayer.values[0][3][0] = 1;
		testLayer.values[0][3][1] = 1;
		testLayer.values[0][3][2] = 0;
		testLayer.values[0][3][3] = 0;
		
		
		
		testLayer.values[1][0][0] = 3;
		testLayer.values[1][0][1] = 2;
		testLayer.values[1][0][2] = 1;
		testLayer.values[1][0][3] = 0;

		testLayer.values[1][1][0] = 1;
		testLayer.values[1][1][1] = 0;
		testLayer.values[1][1][2] = 2;
		testLayer.values[1][1][3] = 0;

		testLayer.values[1][2][0] = 2;
		testLayer.values[1][2][1] = 0;
		testLayer.values[1][2][2] = 0;
		testLayer.values[1][2][3] = 0;
		
		testLayer.values[1][3][0] = 5;
		testLayer.values[1][3][1] = 3;
		testLayer.values[1][3][2] = 0;
		testLayer.values[1][3][3] = 3;
		
		
		testLayer.values[2][0][0] = 0;
		testLayer.values[2][0][1] = 0;
		testLayer.values[2][0][2] = 0;
		testLayer.values[2][0][3] = 0;

		testLayer.values[2][1][0] = 0;
		testLayer.values[2][1][1] = 0;
		testLayer.values[2][1][2] = 0;
		testLayer.values[2][1][3] = 0;

		testLayer.values[2][2][0] = 0;
		testLayer.values[2][2][1] = 0;
		testLayer.values[2][2][2] = 0;
		testLayer.values[2][2][3] = 0;
		
		testLayer.values[2][3][0] = 0;
		testLayer.values[2][3][1] = 0;
		testLayer.values[2][3][2] = 1;
		testLayer.values[2][3][3] = 0;
		

		
		int width = ((testLayer.values[0][0].length -testLayer.Fcollumns + (2* testLayer.pad))/testLayer.step)+ 1;
		
		testOut = new double[testLayer.values.length][width][width];
	}

	@Test
	public void test() {
		testLayer.pool(testLayer.values, testOut, testLayer.step, testLayer.Fcollumns);
		
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
		
		
		for (int i = 0; i < temp.length; i++) {
			for (int j = 0; j < temp[0].length; j++) {
				for (int k = 0; k < temp[0][0].length; k++) {
					assertEquals(temp[i][j][k], testOut[i][j][k], 0); 
				}
			}
		}
	}

}
