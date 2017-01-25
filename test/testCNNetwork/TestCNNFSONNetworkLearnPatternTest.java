//package testCNNetwork;
//
//import static org.junit.Assert.*;
//
//import org.junit.Before;
//import org.junit.Test;
//
//import cnnetwork.FSONNetwork;
//import cnnetwork.Layer;
//
//public class TestCNNFSONNetworkLearnPatternTest {
//
//	FSONNetwork net;
//	@Before
//	public void setUp() throws Exception {
//		net = FSONNetwork.patternNetwork();
//		net.learnPattern();
//	}
//
//	@Test
//	public void test() throws Exception {
//		
//		// Open file for input
//		FSONNetwork.openFileInputBW(net.layers, "testingInput/patterns/24.jpg");
//		// Feed the input through the layers of the network.
//		FSONNetwork.feedForward(net.layers, net.out, false);
//		// Here we are using the sigmoid activation function because 
//		// the output cells are independant of one another. 
//		net.out[0].value = Layer.activationFunction(net.out[0].value);
//		
//		//assertEquals( 1.0, net.out[0].value, 0.2);
//		
//		FSONNetwork.openFileInputBW(net.layers, "testingInput/patterns/25.jpg");
//		// Feed the input through the layers of the network.
//		FSONNetwork.feedForward(net.layers, net.out, false);
//		// Here we are using the sigmoid activation function because 
//		// the output cells are independant of one another. 
//		net.out[0].value = Layer.activationFunction(net.out[0].value);
//		
//		//assertEquals( 1.0, net.out[0].value, 0.2);
//		
//		
//		FSONNetwork.openFileInputBW(net.layers, "testingInput/patterns/17.jpg");
//		// Feed the input through the layers of the network.
//		FSONNetwork.feedForward(net.layers, net.out, false);
//		// Here we are using the sigmoid activation function because 
//		// the output cells are independant of one another. 
//		net.out[0].value = Layer.activationFunction(net.out[0].value);
//		
//		//assertEquals( 1.0, net.out[0].value, 0.2);
//		
//		FSONNetwork.openFileInputBW(net.layers, "testingInput/patterns/34.jpg");
//		// Feed the input through the layers of the network.
//		FSONNetwork.feedForward(net.layers, net.out, false);
//		// Here we are using the sigmoid activation function because 
//		// the output cells are independant of one another. 
//		net.out[0].value = Layer.activationFunction(net.out[0].value);
//		
//		//assertEquals( 0.0, net.out[0].value, 0.2);
//		
//		FSONNetwork.openFileInputBW(net.layers, "testingInput/patterns/23.jpg");
//		// Feed the input through the layers of the network.
//		FSONNetwork.feedForward(net.layers, net.out, false);
//		// Here we are using the sigmoid activation function because 
//		// the output cells are independant of one another. 
//		net.out[0].value = Layer.activationFunction(net.out[0].value);
//		
//		//assertEquals( 0.0, net.out[0].value, 0.2);
//		
//		FSONNetwork.openFileInputBW(net.layers, "testingInput/patterns/28.jpg");
//		// Feed the input through the layers of the network.
//		FSONNetwork.feedForward(net.layers, net.out, false);
//		// Here we are using the sigmoid activation function because 
//		// the output cells are independant of one another. 
//		net.out[0].value = Layer.activationFunction(net.out[0].value);
//		
//		assertEquals( 0.0, net.out[0].value, 0.2);
//		
//		FSONNetwork.openFileInputBW(net.layers, "testingInput/patterns/8.jpg");
//		// Feed the input through the layers of the network.
//		FSONNetwork.feedForward(net.layers, net.out, false);
//		// Here we are using the sigmoid activation function because 
//		// the output cells are independant of one another. 
//		net.out[0].value = Layer.activationFunction(net.out[0].value);
//		
//		//assertEquals( 0.0, net.out[0].value, 0.2);
//		}
//	
//}
