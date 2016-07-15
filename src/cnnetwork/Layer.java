package cnnetwork;

import java.util.LinkedList;

/**
 * This class contains all the code for the structure and function of a neural
 * network, including fully connected, locally connected, max pooling, and
 * convolutional layers.
 * 
 * @author Paula Rudy
 *
 */
public class Layer {

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
		this.filters = new LinkedList<double[][][]>();
		this.type = type;
		this.values = new double[depth][rows][collumns];
	}

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
	public LinkedList<double[][][]> filters;// The list of filters to be applied, in the same order as the list of biases.
	public final LayerType type;// The type of this layer. This determines how the values for the next layer are calculated.
	public double[][][] values;// The actual 3 dimensional array that stores the values for this layer.
	
	/**
	 * This function computes a single output value, given:
	 * 
	 * @param filter
	 *            a three dimensional filter to apply
	 * @param input
	 *            the three dimensional array that contains the value of the
	 *            layer to be used in computation.
	 * @param collumn
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
	 */
	public double compute(double[][][] filter, double[][][] input, int collumn, int row, int depth, double bias) {
		// TODO: add error checking: that filter is not bigger than input, that
		// type is right, that won't go off edge, that everything exists, that
		// filter and bias are same index, that coords are valid
		double result = 0.0;

		for (int i = 0; i < filter.length; i++) {
			for (int j = 0; j < filter[0].length; j++) {
				for (int k = 0; k < filter[0][0].length; k++) {
					result += input[(depth + i)][(row + j)][(collumn + k)] * filter[i][j][k];
				}
			}

		}
		result += bias;
		return result;
	}

	/**
	 * This function computes a single output value as the maximum of the area
	 * of the input selected, given:
	 * 
	 * @param input
	 *            the three dimensional array that contains the value of the
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
	 */
	public double computeMax(double[][][] input, int collumn, int row, int depth, int F) {
		// TODO: add error checking: that kernel size is not bigger than input,
		// that type is right, that won't go off edge, that everything exists,
		// that coords are valid

		double result = 0.0;

		for (int j = 0; ((j < F) && ((row + j) < input[0].length)); j++) {
			for (int k = 0; ((k < F) && ((collumn + k) < input[0][0].length)); k++) {
				if (result < input[depth][(row + j)][(collumn + k)]) {
					result = input[depth][(row + j)][(collumn + k)];
				}
			}
		}

		return result;
	}

	/**
	 * This function is used to compute the convolution of a layer TODO: add
	 * code for padding. add input checking
	 * 
	 * @param input
	 *            the three dimensional array that contains the values of the
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
	 */
	public void convolution(double[][][] input, LinkedList<double[][][]> filters, double[][][] output, int step,
			int padding, LinkedList<Double> biases) {

		// For every filter and bias in the list
		for (int l = 0; l < filters.size(); l++) {

			// Row
			for (int j = 0; j < output[0].length; j += step) {
				// Column
				for (int k = 0; k < output[0][0].length; k += step) {
					output[l][(j / step)][(k / step)] = compute(filters.get(l), input, k, j, 0, biases.get(l));

				}
			}

		}
	}

	/**
	 * This function is used to compute the max pool of a layer TODO: add code
	 * for padding. add input checking
	 * 
	 * @param input
	 *            the three dimensional array that contains the values of the
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
	public void pool(double[][][] input, double[][][] output, int step, int f) {

		// Depth
		for (int l = 0; l < input.length; l++) {
			// Row
			for (int j = 0; (j + f) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + f) <= input[0][0].length; k += step) {
					output[l][(j / step)][(k / step)] = computeMax(input, k, j, l, f);

				}

			}

		}

	}

	/**
	 * This function is used to calculate the locally connected output of a
	 * layer.
	 * 
	 * @param input
	 *            the three dimensional array that contains the values of the
	 *            layer to be used in computation.
	 * @param filters
	 *            the list of three dimensional filters to apply to the input
	 *            layer
	 * @param output
	 *            the three dimensional array to hold the calculated values of
	 *            the local computations.
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 */
	public void local(double[][][] input, LinkedList<double[][][]> filters, double[][][] output, int step, int padding,
			LinkedList<Double> biases) {

		int filterNum = 0;
		// Depth
		for (int l = 0; (l + filters.get(0).length) <= input.length; l++) {
			// Row
			for (int j = 0; (j + filters.get(0)[0].length) <= input[0].length; j += step) {
				// Column
				for (int k = 0; (k + filters.get(0)[0][0].length) <= input[0][0].length; k += step) {
					output[l][(j / step)][(k / step)] = compute(filters.get(filterNum), input, k, j, l,
							biases.get(filterNum));
					filterNum++;
				}

			}

		}

	}

	/**
	 * This function is used to calculate the fully connected output of a layer.
	 * 
	 * @param input
	 *            the three dimensional array that contains the values of the
	 *            layer to be used in computation.
	 * @param filters
	 *            the list of three dimensional filters to apply to the input
	 *            layer
	 * @param output
	 *            the three dimensional array to hold the calculated values of
	 *            the local computations.
	 * @param step
	 *            the "step" of the input layer- the number of columns and rows
	 *            between the filters
	 * @param padding
	 *            the number of zeros to be appended to the rows and columns at
	 *            the edge of the input layer when applying the filters.
	 * @param biases
	 *            the list of biases to be applied to the input layer, in the
	 *            same order as the list of filters.
	 */
	public void full(double[][][] input, LinkedList<double[][][]> filters, double[] output, int step, int padding,
			LinkedList<Double> biases) {
		for (int f = 0; f < filters.size(); f++) {
			output[f] = compute(filters.get(f), input, 0, 0, 0, biases.get(f));
		}
	}
}
