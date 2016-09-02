package testCNNetwork;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Layer;
import cnnetwork.LayerType;

public class TestCNNActivationFunction {

	Layer testLayer;
	
	@Before
	public void setUp() throws Exception {
		//Create and initialize the layer to use for testing
		testLayer = new Layer(3, 3, 3, 2, 2, 3, 4, 1, 0, LayerType.CONV);
	}

	@Test
	public void test() throws Exception {
		double result = testLayer.activationFunction(2);
		assertEquals(result, 0.1192029, 0);
	}

}
