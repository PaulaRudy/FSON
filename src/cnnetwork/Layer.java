package cnnetwork;

import java.util.LinkedList;

import javacalculus.core.CALC;
import javacalculus.core.CalcParser;
import javacalculus.evaluator.CalcSUB;
import javacalculus.struct.CalcDouble;
import javacalculus.struct.CalcObject;
import javacalculus.struct.CalcSymbol;

/**
 * This class contains all the code for the structure and function of a neural
 * network, including fully connected, locally connected, max pooling, and
 * convolutional layers.
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
	public LinkedList<Double> biases;// The list of biases to be applied, in the same order as the list of filters.
	public LinkedList<Filter> filters;// The list of filters to be applied, in the same order as the list of biases.
	public final LayerType type;// The type of this layer. This determines how the values for the next layer are calculated.
	public Cell[][][] cells;// The actual 3 dimensional array that stores the cells for this layer.

	public Layer(int collumns, int rows, int depth, int fcollumns, int frows, int fdepth, int k, int step, int pad,
			LayerType type) {
		super();
		this.collumns = collumns;
		this.rows = rows;
		this.depth = depth;
		this.Fcollumns = fcollumns;
		this.Frows = frows;
		this.Fdepth = fdepth;
		this.K = k;
		this.step = step;
		this.pad = pad;
		this.biases = new LinkedList<Double>();
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
	 * This function computes a single output value, given:
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
	 * @return the newly computed value to be stored into the next layer.
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public double compute(Filter filter, Cell[][][] input, int column, int row, int depth, double bias) throws Exception {
		double result = 0.0;

		//For every cell in the input array, using depth, row, and column as the starting point,
		//multiply that value by the corresponding entry in the filter, and add it to the result
		for (int i = 0; i < filter.weights.length; i++) {
			for (int j = 0; j < filter.weights[0].length; j++) {
				for (int k = 0; k < filter.weights[0][0].length; k++) {
					result += input[(depth + i)][(row + j)][(column + k)].value * filter.weights[i][j][k];
				}
			}

		}
		result += bias;//Add the bias
		
		//Use the activation function to determine actual value (limited between 0 and 1).
		result = activationFunction(result);
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
	 * @return
	 * 			  The found maximum of that area
	 */
	public double computeMax(Cell[][][] input, int collumn, int row, int depth, int F) {

		//Start with 0. Since every value is between 0 and 1,
		//this is guaranteed to be <= to every cell value. 
		double result = 0.0;

		//For every cell at this depth ("depth"), find the max value.
		//Make sure not to go off the edge of the filter "(j < F)",
		//or the input "((row + j) < input[0].length)".
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
	 *            the three dimensional array to hold the calculated values of
	 *            the convolution
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 * @throws Exception 
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public void convolution(Cell[][][] input, LinkedList<Filter> filters, Cell[][][] output, int step,
			int padding, LinkedList<Double> biases) throws Exception {

		// For every filter and bias in the list
		for (int l = 0; l < filters.size(); l++) {

			//Iterate through the filters and input, calling "compute" to
			//apply each filter at the correct location in turn.
			//We are only working at a depth of 0 because the "compute" function 
			//will iterate through the full depth of the filters).

			// Row
			for (int j = 0; (j + filters.get(0).weights[0].length) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + filters.get(0).weights[0][0].length) <= input[0][0].length; k += step) {
					
					output[l][(j / step)][(k / step)].value = compute(filters.get(l), input, k, j, 0, biases.get(l));

					//Record this connection
					filters.get(l).connections.add(new FilterConnection(l, new CellCoord(0, j, k), new CellCoord(l, (j / step), (k / step)), -1));
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
	 *            the three dimensional array to hold the calculated values of
	 *            the max pool
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the sections of input used.
	 * @param f
	 *            the size (f = width = height) of the section of input used in
	 *            the pooling operation.
	 */
	public void pool(Cell[][][] input, LinkedList<Filter> filters, Cell[][][] output, int step, int f) {

		int filterNum = 0;//This is used to iterate over the list of filters (necessary for backpropagation)
		
		//Iterate over the input, calling "computeMax" to pool at each location.
		//Make sure not to go off the edge of the input "((j + f) < input[0].length)".
		//(This check is not needed for depth because "computeMax" only operates on a
		//single depth slice).
		
		// Depth
		for (int l = 0; l < input.length; l++) {
			// Row
			for (int j = 0; (j + f) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + f) <= input[0][0].length; k += step) {
					output[l][(j / step)][(k / step)].value = computeMax(input, k, j, l, f);
					
					//Record this connection
					filters.get(filterNum).connections.add(new FilterConnection(filterNum, new CellCoord(l, j, k), new CellCoord(l, (j / step), (k / step)), -1));
					
					filterNum++;//Make sure to increment this so that you use the next filter each time.

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
	 *            the three dimensional array to hold the calculated values in 
	 *            cells of the local computations.
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 * @throws Exception 
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public void local(Cell[][][] input, LinkedList<Filter> filters, Cell[][][] output, int step, int padding,
			LinkedList<Double> biases) throws Exception {

		int filterNum = 0;//This is used to iterate over the list of filters ("filters") and biases ("biases").
		
		//Iterate over the input, calling "compute" to calculate the result at each location.
		//Make sure not to go off the edge of the input ("(l + filters.get(0).length) <= input.length").
		
		// Depth
		for (int l = 0; (l + filters.get(0).weights.length) <= input.length; l++) {
			// Row
			for (int j = 0; (j + filters.get(0).weights[0].length) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + filters.get(0).weights[0][0].length) <= input[0][0].length; k += step) {
					output[l][(j / step)][(k / step)].value = compute(filters.get(filterNum), input, k, j, l,
							biases.get(filterNum));
					
					//Record this connection
					filters.get(filterNum).connections.add(new FilterConnection(filterNum, new CellCoord(l, j, k), new CellCoord(l, (j / step), (k / step)), -1));
					
					filterNum++;//Make sure to increment this so that you use the next filter and bias each time.
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
	 *            the three dimensional array to hold the calculated values in
	 *            cells of the local computations.
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 * @throws Exception 
	 *             Thrown when the activation function does not return a number
	 *             (see activationFunction()).
	 */
	public void full(Cell[][][] input, LinkedList<Filter> filters, Cell[] output, int step, int padding,
			LinkedList<Double> biases) throws Exception {
		//Apply each filter to the input.
		//Because this is a fully connected layer, each filter is applied to the entire input array,
		// so we do not need to iterate over the input (the "compute" function will iterate through 
		// the full depth of the filter).
		for (int f = 0; f < filters.size(); f++) {
			output[f].value = compute(filters.get(f), input, 0, 0, 0, biases.get(f));
			
			//Record this connection
			filters.get(f).connections.add(new FilterConnection(f, new CellCoord(0, 0, 0), new CellCoord(0, 0, f), -1));
		}
	}
	
	/**
	 * This function is used to ensure that the value for each cell is kept
	 * between 0 and 1
	 * 
	 * @param x
	 *            The original value of the cell as a double
	 * @return The calculated result of the activation function
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 */
	public double activationFunction(double x) throws Exception {
		String function = "1/(1+E^x)";// The actual activation function, in string form

		// Parse the function using JavaCalculus
		CalcParser parser = new CalcParser();
		CalcObject parsed = parser.parse(function);
		CalcObject resultObject = parsed.evaluate();

		// Substitute the passed in value of x
		CalcSymbol symbol = new CalcSymbol("x");
		CalcDouble value = new CalcDouble(x);
		resultObject = CalcSUB.numericSubstitute(resultObject, symbol, value);

		// Evaluate the function using the passed in value of x
		resultObject = CALC.SYM_EVAL(resultObject);

		// Return either the numerical result or throw an exception to indicate
		// it cannot be calculated
		double result;
		if (resultObject.isNumber()) {
			result = Double.parseDouble(resultObject.toString());
		} else {
			throw new Exception();
		}

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
		String function = "(1+E^x)/y";// The actual activation function, in string form

		// Parse the function using JavaCalculus
		CalcParser parser = new CalcParser();
		CalcObject parsed = parser.parse(function);
		CalcObject resultObject = parsed.evaluate();

		// Substitute the passed in value of x
		CalcSymbol symbolx = new CalcSymbol("x");
		CalcDouble valuex = new CalcDouble(x);
		resultObject = CalcSUB.numericSubstitute(resultObject, symbolx, valuex);
		
		// Substitute the passed in value of sumENet
		CalcSymbol symboly = new CalcSymbol("y");
		CalcDouble valuey = new CalcDouble(sumENet);
		resultObject = CalcSUB.numericSubstitute(resultObject, symboly, valuey);

		// Evaluate the function using the passed in value of x
		resultObject = CALC.SYM_EVAL(resultObject);

		// Return either the numerical result or throw an exception to indicate
		// it cannot be calculated
		double result;
		if (resultObject.isNumber()) {
			result = Double.parseDouble(resultObject.toString());
		} else {
			throw new Exception();
		}

		return result;
	}
	
	/**
	 * This is a helper function, used to calculate the sum of e to the power of
	 * the value of each of the values in the input. This value is used in
	 * softmaxActivationFunction as a parameter.
	 * 
	 * @param input
	 *            The input values upon which to calculate the sum.
	 * 
	 * @return The sum e raised to each original value of all the values of the
	 *         input. IE sum(e^xi).
	 */
	public static double sumE(double[] input) {
		
		//Start with 0. This will hold the sum.
		double total = 0;
		
		// Iterate over all the cells. 
		for (int i = 0; i < input.length; i++) {
			total += Math.exp(input[i]);
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
	 *            The input layer's values upon which to calculate the softmax.
	 * @return A one dimensional array of doubles, the same length as the depth
	 *         of the input array, that sum to 1.
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 */
	public static double[] softmax(double[] input) throws Exception {

		double sum = sumE(input);
		double[] output = new double[input.length];
		
		// Iterate over all the cells. Because this is a softmax "layer", it will
		// be one dimensional
		for (int i = 0; i < input.length; i++) {
			output[i] = softmaxActivationFunction(input[i], sum);
		}

		return output;
		
	}

}
