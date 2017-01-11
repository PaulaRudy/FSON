package cnnetwork;

import java.util.LinkedList;

public class Filter {
	public double[][][] weights;// The actual stored weights of this filter
	public double[][][] previousWeights;//Used to store the previous value of the weights to avoid using newly incremented weights before the a full iteration of the backpropagation is finished.
	public double[][][] gradientValues;//The value of the gradient at each location. 
	//Used during backpropagation to store previously computed values.
	//This will always be with respect to the appropriate input because 
	//these values are wiped right before a new input is used.
	//A value of Double.NaN here indicates no gradient value has been computed yet
	//for that location
	//TODO: Are these nessecary?
	public String equationAtFilter;// The equation used to calculate an output for this
							// filter, stored in string form to be used with
							// JavaCalculus
							//TODO Remove. Unnessecary.
	public LinkedList<FilterConnection> connections;// A list of connections that use
												// this filter, to be used in
												// backpropagation.

	public Filter(double[][][] weights, double[][][] previousWeights, double[][][] gradientValues, String equationAtFilter, LinkedList<FilterConnection> connections) {
		this.weights = weights;
		this.previousWeights = previousWeights;
		this.gradientValues = gradientValues;
		this.equationAtFilter = equationAtFilter;
		this.connections = connections;
	}

	/**
	 * Default constructor for a filter with values.
	 * Initializes the equation to an empty string and sets up 
	 * the linked list.
	 * @param weights 
	 * 			A 3-dimensional array of doubles that are the actual
	 * 			stored weights of this filter
	 */
	public Filter(double[][][] weights) {
		this.weights = weights;
		this.previousWeights = new double[weights.length][weights[0].length][weights[0][0].length];
		
		//Make a deep copy of the weights
		// Depth
		for (int l = 0; l < weights.length; l++) {
			// Row
			for (int m = 0; m < weights[0].length; m++) {
				// Column
				for (int n = 0; n < weights[0][0].length; n++) {
					this.previousWeights[l][m][n] = this.weights[l][m][n];
				}
			}
		}
				
		this.gradientValues = new double[weights.length][weights[0].length][weights[0][0].length];
		
		// Because java initializes arrays of doubles to 0, we need to set them
		// to Double.NaN to indicate no gradient has been calculated yet.
		// Depth
		for (int l = 0; l < weights.length; l++) {
			// Row
			for (int m = 0; m < weights[0].length; m++) {
				// Column
				for (int n = 0; n < weights[0][0].length; n++) {
					this.gradientValues[l][m][n] = Double.NaN;
				}
			}
		}
				
		this.equationAtFilter = "";
		this.connections = new LinkedList<FilterConnection>();
	}
	
	//TODO heading
	public Filter(int depth, int rows, int cols) {
		this.weights = new double[depth][rows][cols];
		this.previousWeights = new double[depth][rows][cols];
		this.gradientValues = new double[depth][rows][cols];;
		
		// Because java initializes arrays of doubles to 0, we need to set them
		// to Double.NaN to indicate no gradient has been calculated yet.
		// Depth
		for (int l = 0; l < weights.length; l++) {
			// Row
			for (int m = 0; m < weights[0].length; m++) {
				// Column
				for (int n = 0; n < weights[0][0].length; n++) {
					this.gradientValues[l][m][n] = Double.NaN;
				}
			}
		}
				
		this.equationAtFilter = "";
		this.connections = new LinkedList<FilterConnection>();
	}

}
