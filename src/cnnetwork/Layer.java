package cnnetwork;

import java.util.LinkedList;

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

	public final int collumns;
	public final int rows;
	public final int depth;
	public final int Fcollumns;
	public final int Frows;
	public final int Fdepth;
	public int K;
	public int step;
	public int pad;
	public LinkedList<Double> biases;
	public LinkedList<double[][][]> filters;
	public final LayerType type;
	public double[][][] values;

	public static void main(String[] args) {

	}

	/**
	 * 
	 * @param filter
	 * @param input
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
	 * @return
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
	 * 
	 * @param input
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
	 *            dimension (F x F x (full depth)) of section of input to find
	 *            max in
	 * @return
	 */
	public double computeMax(double[][][] input, int collumn, int row, int depth, int F) {
		// TODO: add error checking: that kernel size is not bigger than input,
		// that type is right, that won't go off edge, that everything exists,
		// that coords are valid

		double result = 0.0;

		for (int j = 0; j < F; j++) {
			for (int k = 0; k < F; k++) {
				if (result < input[depth][(row + j)][(collumn + k)]) {
					result = input[depth][(row + j)][(collumn + k)];
				}
			}
		}

		return result;
	}
	
	/**
	 * TODO: add code for padding. add input checking
	 * @param input
	 * @param filters
	 * @param output
	 * @param step
	 * @param padding
	 * @param biases
	 */
	public void convolution(double [][][] input, LinkedList<double[][][]> filters, double [][][] output, int step, int padding, LinkedList<Double> biases){
	
		//For every filter and bias in the list
		for(int l = 0; l< filters.size(); l++){
			
				//Row
				for (int j = 0; j < output[0].length; j+=step) {
					//Column
					for (int k = 0; k < output[0][0].length; k+=step) {
						output[l][(j/step)][(k/step)] = compute(filters.get(l), input, k, j,0, biases.get(l));
						
					}
				}

		}
	}
	
	/**
	 * TODO: add code for padding. add input checking
	 * @param input
	 * @param filters
	 * @param output
	 * @param step
	 * @param padding
	 * @param biases
	 */
	public void pool(double [][][] input, double [][][] output, int step, int f){

		// Depth
		for (int l = 0; l < input.length; l++){
			// Row
			for (int j = 0; j < input[0].length; j += step) {
				// Column
				for (int k = 0; k < input[0][0].length; k += step) {
					output[l][(j / step)][(k / step)] = computeMax(input, k, j, l, f);

				}

			}
			
		}
		
	}
	
	/**
	 * 
	 * @param input
	 * @param filters
	 * @param output
	 * @param step
	 * @param padding
	 * @param biases
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
}
