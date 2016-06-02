package cnnetwork;

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