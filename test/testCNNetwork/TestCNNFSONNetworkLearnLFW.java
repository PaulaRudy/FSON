package testCNNetwork;

import org.junit.Before;
import org.junit.Test;

import cnnetwork.FSONNetwork;

public class TestCNNFSONNetworkLearnLFW {

	FSONNetwork net;
	@Before
	public void setUp() throws Exception {
		net = FSONNetwork.sampleNetwork();
	}

	@Test
	public void test() throws Exception {
		net.learnLFW();
	}

}
