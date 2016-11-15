package cnnetwork;

import java.util.LinkedList;

public class Filter {
	public double[][][] weights;// The actual stored weights of this filter
	public double[][][] gradientValues;//The value of the gradient at each location. 
	//Used during backpropagation to store previously computed values.
	//This will always be with respect to the appropriate weight because 
	//weights are calculated layer by layer and are overwritten before 
	//being used.
	//A value of -1 here indicates no gradient value has been computed yet
	//for that location
	public String equationAtFilter;// The equation used to calculate an output for this
							// filter, stored in string form to be used with
							// JavaCalculus
	public LinkedList<FilterConnection> connections;// A list of connections that use
												// this filter, to be used in
												// backpropagation.

	public Filter(double[][][] weights, double[][][] gradientValues, String equationAtFilter, LinkedList<FilterConnection> connections) {
		this.weights = weights;
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
		this.gradientValues = new double[weights.length][weights[0].length][weights[0][0].length];
		
		// Because java initializes arrays of doubles to 0, we need to set them
		// to -1 to indicate no gradient has been calculated yet.
		// Depth
		for (int l = 0; l < weights.length; l++) {
			// Row
			for (int m = 0; m < weights[0].length; m++) {
				// Column
				for (int n = 0; n < weights[0][0].length; n++) {
					this.gradientValues[l][m][n] = -1;
				}
			}
		}
				
		this.equationAtFilter = "";
		this.connections = new LinkedList<FilterConnection>();
	}
	
	//TODO
	public Filter(int depth, int rows, int cols) {
		this.weights = new double[depth][rows][cols];
		this.gradientValues = new double[depth][rows][cols];;
		
		// Because java initializes arrays of doubles to 0, we need to set them
		// to -1 to indicate no gradient has been calculated yet.
		// Depth
		for (int l = 0; l < weights.length; l++) {
			// Row
			for (int m = 0; m < weights[0].length; m++) {
				// Column
				for (int n = 0; n < weights[0][0].length; n++) {
					this.gradientValues[l][m][n] = -1;
				}
			}
		}
				
		this.equationAtFilter = "";
		this.connections = new LinkedList<FilterConnection>();
	}

}
