package cnnetwork;

import java.util.LinkedList;

public class Filter {
	public double[][][] weights;// The actual stored weights of this filter
	public String equationAtFilter;// The equation used to calculate an output for this
							// filter, stored in string form to be used with
							// JavaCalculus
	public LinkedList<FilterConnection> connections;// A list of connections that use
												// this filter, to be used in
												// backpropagation.

	public Filter(double[][][] weights, String equationAtFilter, LinkedList<FilterConnection> connections) {
		this.weights = weights;
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
		this.equationAtFilter = "";
		this.connections = new LinkedList<FilterConnection>();
	}

}
