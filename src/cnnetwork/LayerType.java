package cnnetwork;

/**
 * An enum used to indicate the type of a Layer:
 * Convolutional (CONV),
 * Maxpooling (MAXPOOL),
 * Locally connected (LOCAL), 
 * Fully connected (FULLY),
 * or
 * Softmax (SOFTMAX).
 */
public enum LayerType {
	CONV(1), MAXPOOL(2), LOCAL(3), FULLY(4), SOFTMAX(5);

	private int value;

	private LayerType(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}
	
};