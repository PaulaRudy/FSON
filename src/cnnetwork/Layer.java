package cnnetwork;

import java.util.LinkedList;

/**
 * This class contains all the code for the structure and function of a neural
 * network, including fully connected, locally connected, max pooling, and
 * convolutional layers, as well as activation functions and their helpers.
 * 
 * @author Paula Rudy
 *
 */
public class Layer {

	public final int collumns;// The width of the layer
	public final int rows;// The height of the layer
	public final int depth;// the depth of the layer
	public final int Fcollumns;// The width of the filters for the layer
	public final int Frows;// The height of the filters for this layer
	public final int Fdepth;// The depth of the filters for this layer
	public int K;// The number of filters to be applied to this layer
	public int step;// The "step" of the layer- the number of columns and rows between the filters
	public int pad;// The number of zeros appended to the rows and columns at the edge of the layer when applying the filters.
	public LinkedList<Cell> biases;// The list of biases to be applied, in the same order as the list of filters.
	public LinkedList<Filter> filters;// The list of filters to be applied, in the same order as the list of biases.
	public final LayerType type;// The type of this layer. This determines how the values for the next layer are calculated.
	public Cell[][][] cells;// The actual 3 dimensional array that stores the cells for this layer.

	public Layer(int collumns, int rows, int depth, int fcollumns, int frows, int fdepth, int k, int step, int pad,
			LayerType type) {
		this.collumns = collumns;
		this.rows = rows;
		this.depth = depth;
		this.Fcollumns = fcollumns;
		this.Frows = frows;
		this.Fdepth = fdepth;
		this.K = k;
		this.step = step;
		this.pad = pad;
		this.biases = new LinkedList<Cell>();
		this.filters = new LinkedList<Filter>();
		this.type = type;
		this.cells = new Cell[depth][rows][collumns];

		// Initialize the cells because java won't do it for you
		// Depth
		for (int l = 0; l < depth; l++) {
			// Row
			for (int m = 0; m < rows; m++) {
				// Column
				for (int n = 0; n < collumns; n++) {
					this.cells[l][m][n] = new Cell();
				}
			}
		}
	}

	/**
	 * This function is used to initialize all the filters and biases for this
	 * layer. All filter weights are initialized to 0.5, and all biases to 0.
	 */
	public void initLayer() {
		// Create and initialize the filters
		for (int i = 0; i < this.K; i++) {
			// Create the filter weights
			double[][][] newFilterWeights = new double[this.Fdepth][this.Frows][this.Fcollumns];
			for (int x = 0; x < this.Fdepth; x++) {
				for (int y = 0; y < this.Frows; y++) {
					for (int z = 0; z < this.Fcollumns; z++) {
						newFilterWeights[x][y][z] = 0.5;
					}
				}
			}

			Filter newFilter = new Filter(newFilterWeights);// Use the default constructor with the newly created filter weights
			this.filters.add(newFilter);// Actually add the filter to the list of filters in the layer
			Cell newBias = new Cell(0.0);// Create the bias for this filter
			this.biases.add(newBias);// Add the bias to the list of biases in the layer.
		}
	}

	/**
	 * This function computes a single output value. Previous values are used
	 * if any of the input values have already been updated in this session
	 * of backpropagation (if that paticular value has a partial derivative 
	 * already stored).
	 * 
	 * @param filter
	 *            a three dimensional filter to apply
	 * @param input
	 *            the three dimensional array that contains the cells of the
	 *            layer to be used in computation.
	 * @param column
	 *            [][][x] location of top left coordinate of input to apply
	 *            filter to
	 * @param row
	 *            [][x][] location of top left coordinate of input to apply
	 *            filter to
	 * @param depth
	 *            [x][][] location of top left coordinate of input to apply
	 *            filter to
	 * @param bias
	 *            the bias to be used in this computation.
	 * @param lastLayer
	 *            A boolean indicating if this computation's result will be used
	 *            for the last layer or not. A "true" value here indicates that
	 *            this computation's result will be used to calculate the values
	 *            of the "out" layer, and so should not use the activation
	 *            function.
	 * @return the newly computed value to be stored into the next layer.
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public static double compute(Filter filter, Cell[][][] input, int column, int row, int depth, Cell bias,
			boolean lastLayer) throws Exception {
		double result = 0.0;

		// For every cell in the input array, using depth, row, and column as
		// the starting point,
		// multiply that value by the corresponding entry in the filter,
		// and add it to the result
		for (int i = 0; i < filter.weights.length; i++) {
			for (int j = 0; j < filter.weights[0].length; j++) {
				for (int k = 0; k < filter.weights[0][0].length; k++) {

					// If the filter's weight at this position has been updated during this iteration...
					if (!Double.isNaN(filter.gradientValues[i][j][k])){
						//... use the previous version.
						result += input[(depth + i)][(row + j)][(column + k)].value * filter.previousWeights[i][j][k];
					} else {
						result += input[(depth + i)][(row + j)][(column + k)].value * filter.weights[i][j][k];
					}
				}
			}

		}
		
		// Add the bias
		// If the bias's value has be updated during this iteration...
		if (!Double.isNaN(bias.derivative)){
			// ... use the previous version.
			result+= bias.previousValue;
		} else {
			result += bias.value;// Add the bias
		}

		// If the result of this computation isn't going to be stored in the
		// output of this network (if the input array is not from the last layer
		// before "out")...
		if (!lastLayer) {
			// Use the activation function to determine actual value (limited between 0 and 1).
			result = activationFunction(result);
		}

		return result;
	}

	/**
	 * This function computes a single output value as the maximum of the area
	 * of the input selected (only at a single depth slice), given:
	 * 
	 * @param input
	 *            the three dimensional array that contains the cells of the
	 *            layer to be used in computation.
	 * @param collumn
	 *            [][][x] location of top left coordinate of input to find max
	 *            in
	 * @param row
	 *            [][x][] location of top left coordinate of input to find max
	 *            in
	 * @param depth
	 *            [x][][] location of top left coordinate of input to find max
	 *            in
	 * @param F
	 *            dimension (F x F (@ the depth)) of section of input to find
	 *            max in
	 * @return The found maximum of that area
	 */
	public double computeMax(Cell[][][] input, int collumn, int row, int depth, int F) {

		// Start with 0. Since every value is between 0 and 1,
		// this is guaranteed to be <= to every cell value.
		double result = 0.0;

		// For every cell at this depth ("depth"), find the max value.
		// Make sure not to go off the edge of the filter:
		//		 "(j < F)"
		// or the input:
		//		 "((row + j) < input[0].length)"
		for (int j = 0; ((j < F) && ((row + j) < input[0].length)); j++) {
			for (int k = 0; ((k < F) && ((collumn + k) < input[0][0].length)); k++) {
				if (result < input[depth][(row + j)][(collumn + k)].value) {
					result = input[depth][(row + j)][(collumn + k)].value;
				}
			}
		}

		return result;
	}

	/**
	 * This function is used to compute the convolution of a layer.
	 * 
	 * @param input
	 *            the three dimensional array that contains the cells of the
	 *            layer to be used in computation.
	 * @param filters
	 *            the list of three dimensional filters to apply to the input
	 *            layer
	 * @param output
	 *            the three dimensional array of Cells to hold the calculated
	 *            values of the convolution
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 * @param store
	 *            An indication if this network should be set up or not. A
	 *            "true" value here means that connections should be recorded
	 *            because this is the first pass through the network and the
	 *            structure needs to be recorded. A false value here means this
	 *            is network is already set up and does not need to record
	 *            connections (no connections will be recorded in this call).
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public void convolution(Cell[][][] input, LinkedList<Filter> filters, Cell[][][] output, int step, int padding,
			LinkedList<Cell> biases, boolean store) throws Exception {

		// For every filter and bias in the list
		for (int l = 0; l < filters.size(); l++) {

			// Iterate through the filters and input, calling "compute" to
			// apply each filter at the correct location in turn.
			// We are only working at a depth of 0 because the "compute"
			// function will iterate through the full depth of the filters).

			// Row
			for (int j = 0; (j + filters.get(0).weights[0].length) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + filters.get(0).weights[0][0].length) <= input[0][0].length; k += step) {

					output[l][(j / step)][(k / step)].value = compute(filters.get(l), input, k, j, 0, biases.get(l),
							false);

					// If this is a new network...
					if (store) {
						// Record this connection. This information will be used during backpropagation.
						filters.get(l).connections.add(new FilterConnection(l, new CellCoord(0, j, k),
								new CellCoord(l, (j / step), (k / step))));
					}
				}
			}

		}
	}

	/**
	 * This function is used to compute the max pool of a layer.
	 * 
	 * @param input
	 *            the three dimensional array that contains the cells of the
	 *            layer to be used in computation.
	 * @param output
	 *            the three dimensional array of Cells to hold the calculated
	 *            values of the max pool
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the sections of input used.
	 * @param store
	 *            An indication if this network should be set up or not. A
	 *            "true" value here means that connections should be recorded
	 *            because this is the first pass through the network and the
	 *            structure needs to be recorded. A false value here means this
	 *            is network is already set up and does not need to record
	 *            connections (no connections will be recorded in this call).
	 * @param f
	 *            the size (f = width = height) of the section of input used in
	 *            the pooling operation.
	 */
	public void pool(Cell[][][] input, LinkedList<Filter> filters, Cell[][][] output, int step, int f, boolean store) {

		int filterNum = 0;// This is used to iterate over the list of filters necessary for backpropagation)

		// Iterate over the input, calling "computeMax" to pool at each location.
		// Make sure not to go off the edge of the input 
		// "((j + f) < input[0].length)".
		// (This check is not needed for depth because "computeMax" only
		// operates on a single depth slice).

		// Depth
		for (int l = 0; l < input.length; l++) {
			// Row
			for (int j = 0; (j + f) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + f) <= input[0][0].length; k += step) {
					output[l][(j / step)][(k / step)].value = computeMax(input, k, j, l, f);

					// If this is a new network...
					if (store) {
						// Record this connection. This information will be used during backpropagation.
						filters.get(filterNum).connections.add(new FilterConnection(filterNum, new CellCoord(l, j, k),
								new CellCoord(l, (j / step), (k / step))));
					}

					filterNum++;// Make sure to increment this so that you use the next filter each time.

				}

			}

		}

	}

	/**
	 * This function is used to calculate the locally connected output of a
	 * layer.
	 * 
	 * @param input
	 *            the three dimensional array that contains the cells of the
	 *            layer to be used in computation.
	 * @param filters
	 *            the list of three dimensional filters to apply to the input
	 *            layer
	 * @param output
	 *            the three dimensional array of Cells to hold the calculated
	 *            values in cells of the local computations.
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 * @param store
	 *            An indication if this network should be set up or not. A
	 *            "true" value here means that connections should be recorded
	 *            because this is the first pass through the network and the
	 *            structure needs to be recorded. A false value here means this
	 *            is network is already set up and does not need to record
	 *            connections (no connections will be recorded in this call).
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public void local(Cell[][][] input, LinkedList<Filter> filters, Cell[][][] output, int step, int padding,
			LinkedList<Cell> biases, boolean store) throws Exception {

		int filterNum = 0;// This is used to iterate over the list of filters ("filters") and biases ("biases").

		// Iterate over the input, calling "compute" to calculate the result at each location.
		// Make sure not to go off the edge of the input: 
		// "(l + filters.get(0).length) <= input.length"

		// Depth
		for (int l = 0; (l + filters.get(0).weights.length) <= input.length; l++) {
			// Row
			for (int j = 0; (j + filters.get(0).weights[0].length) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + filters.get(0).weights[0][0].length) <= input[0][0].length; k += step) {
					output[l][(j / step)][(k / step)].value = compute(filters.get(filterNum), input, k, j, l,
							biases.get(filterNum), false);

					// If this is a new network...
					if (store) {
						// Record this connection. This information will be used during backpropagation.
						filters.get(filterNum).connections.add(new FilterConnection(filterNum, new CellCoord(l, j, k),
								new CellCoord(l, (j / step), (k / step))));
					}

					filterNum++;// Make sure to increment this so that you use the next filter and bias each time.
				}

			}

		}

	}

	/**
	 * This function is used to calculate the fully connected output of a layer.
	 * 
	 * @param input
	 *            the three dimensional array that contains the cells of the
	 *            layer to be used in computation.
	 * @param filters
	 *            the list of three dimensional filters to apply to the input
	 *            layer
	 * @param output
	 *            the three dimensional array of Cells to hold the calculated
	 *            values in cells of the local computations.
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 * @param store
	 *            An indication if this network should be set up or not. A
	 *            "true" value here means that connections should be recorded
	 *            because this is the first pass through the network and the
	 *            structure needs to be recorded. A false value here means this
	 *            is network is already set up and does not need to record
	 *            connections (no connections will be recorded in this call).
	 * @param lastLayer
	 *            A boolean indicating if this computation's result will be used
	 *            for the last layer or not. A "true" value here indicates that
	 *            this computation's result will be used to calculate the values
	 *            of the "out" layer, and so the activation function should not
	 *            be used.
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public void full(Cell[][][] input, LinkedList<Filter> filters, Cell[] output, int step, int padding,
			LinkedList<Cell> biases, boolean store, boolean lastLayer) throws Exception {
		// Apply each filter to the input.
		// Because this is a fully connected layer, each filter is applied to the entire input array,
		// so we do not need to iterate over the input 
		// (the "compute" function will iterate through the full depth of the filter).
		for (int f = 0; f < filters.size(); f++) {
			output[f].value = compute(filters.get(f), input, 0, 0, 0, biases.get(f), lastLayer);

			// If this is a new network...
			if (store) {
				// Record this connection. This information will be used during backpropagation.
				filters.get(f).connections.add(new FilterConnection(f, new CellCoord(0, 0, 0), new CellCoord(0, 0, f)));
			}
		}
	}

	/**
	 * This function is used to ensure that the value for each cell is kept
	 * between 0 and 1. It is a "sigmoid" function.
	 * 
	 * @param x
	 *            The original value of the cell as a double
	 * @return The calculated result of the activation function
	 */
	public static double activationFunction(double x) {
		
		double denom = 1 + (Math.exp(0.0-x));
		double result = 1/ denom;

		return result;
	}

	/**
	 * This function is a helper function to softmax() to calculate the value of
	 * a single cell in a k way softmax.
	 * 
	 * @param x
	 *            The original value of the cell as a double
	 * @param sumENet
	 *            The sum e raised to each original value of all the cells of
	 *            the input. IE sum(e^xi). Use the function sumE() to obtain.
	 * @return The calculated result of the activation function
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 */
	public static double softmaxActivationFunction(double x, double sumENet) throws Exception {
		
		if (sumENet == Double.POSITIVE_INFINITY){
			return 0;
		} else if(sumENet == 0){
			return Double.POSITIVE_INFINITY;
		}else {
			return (Math.exp(x) / sumENet);
		}
	}

	/**
	 * This is a helper function, used to calculate the sum of e to the power of
	 * the value of each of the values in the input. This value is used in
	 * softmaxActivationFunction as a parameter.
	 * 
	 * @param input
	 *            The cells containing the input values upon which to calculate
	 *            the sum.
	 * 
	 * @return The sum e raised to each original value of all the values of the
	 *         input. IE sum(e^xi).
	 */
	public static double sumE(Cell[] input) {

		// Start with 0. This will hold the sum.
		double total = 0;

		// This will hold the max value in the input array
		double max = input[0].value;

		// Iterate over all the cells to find the max value
		for (int i = 0; i < input.length; i++) {
			if (input[i].value > max) {
				max = input[i].value;
			}
		}

		// Iterate over all the cells.
		// While you do so, and before you apply the exponent,
		// move all the datapoints to put the maximum value of the input at 0.
		// This is to avoid problems with input that is made up of entirely
		// very large or very small numbers, or data has both - and + numbers
		// (because the "detail" in the dataset will be lost without treatment).
		for (int i = 0; i < input.length; i++) {
			input[i].value -= max;
			total += Math.exp(input[i].value);
		}

		return total;
	}

	/**
	 * This function is used to calculate a k way softmax- an array of values
	 * where all the values sum to 1. This is used in the final step in a k-way
	 * classification; the output values represent the probability that the
	 * input represents that particular class.
	 * 
	 * @param input
	 *            The input layer's cells upon which to calculate the softmax.
	 * @return A one dimensional array of Cells, the same length as the depth of
	 *         the input array, that sum to 1.
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 */
	public static void softmax(Cell[] input) throws Exception {

		double sum = sumE(input);

		// Iterate over all the cells. Because this is a softmax "layer", it will be one dimensional
		for (int i = 0; i < input.length; i++) {
			input[i].value = softmaxActivationFunction(input[i].value, sum);
			// Because this function is only used for processing BEFORE
			// backpropagation, we don't need to worry about any stored
			// derivatives being preserved.
			input[i].derivative = Double.NaN;
		}
		
		// If all the input is the same, then it should be equal to 1/n
		// where n is the number of inputs. 
		// This is sometimes not the case with very large or very small numbers, 
		// (see: the application of infinity in the softmaxActivationFunction
		// defined in this file)
		// ...hence this work around
		if ((input.length > 1) && (input[0].value == input[1].value)){
			int c;
			for (c = 2; ((c < input.length)&& (input[c].value == input[0].value)); c++) {
				
			}
			if (c  == input.length){
				for (int i = 0; i < input.length; i++) {
					input[i].value = 1.0/input.length;
				}
			}
		}

	}

}
