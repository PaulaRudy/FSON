package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;
import cnnetwork.NetworkCell;

/**
 * This tests the "full" function found in cnnetwork.Layer.java
 * 
 */
public class TestCNNFullFunction {

	Layer testLayer;

	double[][][] testFilter0, testFilter1, testFilter2, testFilter3;

	double testBias0, testBias1, testBias2, testBias3;

	NetworkCell[] testOut;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		//Create and initialize the layer to use for testing
		testLayer = new Layer(3, 3, 3, 3, 3, 3, 3, 1, 0, LayerType.CONV);

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
		testFilter0[0][0][2] = 0;

		testFilter0[0][1][0] = 0;
		testFilter0[0][1][1] = 1;
		testFilter0[0][1][2] = 0;

		testFilter0[0][2][0] = 1;
		testFilter0[0][2][1] = 0;
		testFilter0[0][2][2] = 0;

		testFilter0[1][0][0] = 1;
		testFilter0[1][0][1] = -1;
		testFilter0[1][0][2] = 1;

		testFilter0[1][1][0] = 0;
		testFilter0[1][1][1] = -1;
		testFilter0[1][1][2] = 0;

		testFilter0[1][2][0] = 0;
		testFilter0[1][2][1] = -1;
		testFilter0[1][2][2] = 0;

		testFilter0[2][0][0] = -1;
		testFilter0[2][0][1] = 1;
		testFilter0[2][0][2] = 0;

		testFilter0[2][1][0] = -1;
		testFilter0[2][1][1] = 1;
		testFilter0[2][1][2] = 0;

		testFilter0[2][2][0] = -1;
		testFilter0[2][2][1] = 1;
		testFilter0[2][2][2] = 0;

		testLayer.filters.add(testFilter0);//Add the filter to the list of filters in the layer

		testBias0 = 1;//Create a bias

		testLayer.biases.add(testBias0);//Add the bias to the list of biases in the layer

		//Create and initialize the second filter
		testFilter1 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter1[0][0][0] = -1;
		testFilter1[0][0][1] = -1;
		testFilter1[0][0][2] = -1;

		testFilter1[0][1][0] = 1;
		testFilter1[0][1][1] = 0;
		testFilter1[0][1][2] = 1;

		testFilter1[0][2][0] = 1;
		testFilter1[0][2][1] = 1;
		testFilter1[0][2][2] = 1;

		testFilter1[1][0][0] = -1;
		testFilter1[1][0][1] = -1;
		testFilter1[1][0][2] = -1;

		testFilter1[1][1][0] = -1;
		testFilter1[1][1][1] = 1;
		testFilter1[1][1][2] = 0;

		testFilter1[1][2][0] = -1;
		testFilter1[1][2][1] = 1;
		testFilter1[1][2][2] = 1;

		testFilter1[2][0][0] = 0;
		testFilter1[2][0][1] = 0;
		testFilter1[2][0][2] = 0;

		testFilter1[2][1][0] = 0;
		testFilter1[2][1][1] = -1;
		testFilter1[2][1][2] = -1;

		testFilter1[2][2][0] = 0;
		testFilter1[2][2][1] = -1;
		testFilter1[2][2][2] = -1;

		testLayer.filters.add(testFilter1);//Add the filter to the list of filters in the layer

		testBias1 = 0;//Create a bias

		testLayer.biases.add(testBias1);//Add the bias to the list of biases in the layer

		//Create and initialize the third filter
		testFilter2 = new double[testLayer.Fdepth][testLayer.Frows][testLayer.Fcollumns];

		testFilter2[0][0][0] = 0;
		testFilter2[0][0][1] = 1;
		testFilter2[0][0][2] = 2;

		testFilter2[0][1][0] = 1;
		testFilter2[0][1][1] = 0;
		testFilter2[0][1][2] = 1;

		testFilter2[0][2][0] = 2;
		testFilter2[0][2][1] = 1;
		testFilter2[0][2][2] = 0;

		testFilter2[1][0][0] = 1;
		testFilter2[1][0][1] = -1;
		testFilter2[1][0][2] = 0;

		testFilter2[1][1][0] = -1;
		testFilter2[1][1][1] = 1;
		testFilter2[1][1][2] = -1;

		testFilter2[1][2][0] = 0;
		testFilter2[1][2][1] = -1;
		testFilter2[1][2][2] = 1;

		testFilter2[2][0][0] = -1;
		testFilter2[2][0][1] = 1;
		testFilter2[2][0][2] = 2;

		testFilter2[2][1][0] = 1;
		testFilter2[2][1][1] = 2;
		testFilter2[2][1][2] = 1;

		testFilter2[2][2][0] = 2;
		testFilter2[2][2][1] = 1;
		testFilter2[2][2][2] = -1;

		testLayer.filters.add(testFilter2);//Add the filter to the list of filters in the layer

		testBias2 = -1;//Create a bias

		testLayer.biases.add(testBias2);//Add the bias to the list of biases in the layer

		//Create and initialize the array to use to store the output
		testOut = new NetworkCell[testLayer.filters.size()];
		
		//Don't forget to initialize the cells- java won't do it for you.
		for (int d = 0; d < testLayer.filters.size(); d++) {
			
					testOut[d] = new NetworkCell();

		}
	}

	@Test
	public void test() {
		testLayer.full(testLayer.cells, testLayer.filters, testOut, 1, 0, testLayer.biases);

		//This is an array of values we expect to see in testOut
		double[] temp = new double[3];

		temp[0] = 7;
		temp[1] = -4;
		temp[2] = 20;

		//Compare the temp array to the values stored in the output array
		for (int i = 0; i < temp.length; i++) {
			assertEquals(temp[i], testOut[i].value, 0);

		}
	}

}
