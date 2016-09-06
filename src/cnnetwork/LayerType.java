package cnnetwork;

/**
 * An enum used to indicate the type of a Layer:
 * Convolutional (CONV),
 * Maxpooling (MAXPOOL),
 * Locally connected (LOCAL), 
 * or
 * Fully connected (FULLY).
 */
public enum LayerType {
	CONV(1), MAXPOOL(2), LOCAL(3), FULLY(4);

	private int value;

	private LayerType(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}
	
};