package testCNNetwork;

import static org.junit.Assert.*;

import java.util.LinkedList;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.Cell;
import cnnetwork.FSONNetwork;
import cnnetwork.Layer;
import cnnetwork.LayerType;

/**
 * This tests the "computePartialDerivative" function(s) in
 * cnnetwork.FSONNetwork.java
 * 
 */
public class TestCNNComputePartialDerivative {
	LinkedList<Layer> layers;
	Cell[] out;
	double[] expect;

	@Before
	public void setUp() throws Exception {

		// Create the list of layers that will make up our network
		layers = new LinkedList<Layer>();

		// Create and initialize the first layer
		Layer l1 = new Layer(3, 3, 1, 2, 2, 1, 1, 1, 0, LayerType.CONV);
		l1.initLayer();

		// Add the first layer to the list
		layers.add(0, l1);

		// Create and initialize the second layer
		Layer l2 = new Layer(2, 2, 1, 2, 2, 1, 3, 1, 0, LayerType.FULLY);
		l2.initLayer();

		// Add the second layer to the list
		layers.add(1, l2);

		// Create and initialize the array of Cell that will hold the output of
		// this network.
		out = new Cell[3];
		out[0] = new Cell();
		out[1] = new Cell();
		out[2] = new Cell();

		// Do a forward pass through the network to set up the connections
		// needed for backpropagation
		layers.get(0).convolution(layers.get(0).cells, layers.get(0).filters, layers.get(1).cells, layers.get(0).step,
				layers.get(0).pad, layers.get(0).biases);
		layers.get(1).full(layers.get(1).cells, layers.get(1).filters, out, layers.get(1).step, layers.get(1).pad,
				layers.get(1).biases);
		Layer.softmax(out);

		// Set up the array that will hold the expected values of the cells in
		// "out".
		expect = new double[3];
		expect[0] = 0;
		expect[1] = 0;
		expect[2] = 1;
	}

	/**
	 * @throws java.lang.Exception
	 */
	@Test
	public void test() throws Exception {

		// Test the calculated partial derivative of the first weight of the
		// first filter of the last layer
		// (IE Layer 2, Filter 1, coordinates [0][0][0]).
		double testResultWeight5 = FSONNetwork.computePartialDerivative(layers, out, 1, 0, 0, 0, 0, expect);
		assertEquals(0.01763939922793867, testResultWeight5, 0.000000001);

		// Test the calculated partial derivative of the first weight of the
		// first filter of the first layer
		// (IE Layer 1, Filter 1, coordinates [0][0][0]).
		double testResultWeight1 = FSONNetwork.computePartialDerivative(layers, out, 0, 0, 0, 0, 0, expect);
		assertEquals(0.0, testResultWeight1, 0.000000001);

	}

}
