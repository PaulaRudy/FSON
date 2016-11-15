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
	
	public static LayerType fromString(String type){
	     switch (type) {
         case "CONV":
         case "1":
             return LayerType.CONV;
         case "MAXPOOL":
         case "2":
        	 return LayerType.MAXPOOL;
         case "LOCAL":
         case "3":
             return LayerType.LOCAL;
		case "FULLY":
         case "4":
        	 return LayerType.FULLY;
         default:
             throw new IllegalArgumentException("Invalid LayerType: " + type);
     }
	}
	
};