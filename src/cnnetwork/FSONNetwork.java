package cnnetwork;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

/**
 * This class contains all the functions necesscary to implement a full network,
 * including fuctions for learning and dealing with input, as well as a sample
 * implementation of a FSONNetwork.
 * 
 * @author Paula Rudy
 *
 */
public class FSONNetwork {

	public LinkedList<Layer> layers; // The layers that make up this network
	public Cell[] out; // This is the last "layer" of this network, the "output".
	public String saveFile; // This is the filename of the text file used to store this network's information for recovery purposes if learning is interupted.
	
	public FSONNetwork(LinkedList<Layer> layers, Cell[] out, String saveFile) {
		this.layers = layers;
		this.out = out;
		this.saveFile = saveFile;
	}

	public FSONNetwork(LinkedList<Layer> layers, Cell[] out) {
		this.layers = layers;
		this.out = out;
		this.saveFile = null;
	}
	
	public FSONNetwork() {
		this.layers = new  LinkedList<Layer>();
		this.out = null;
		this.saveFile = null;
	}
	
	/**
	 * This function creates and sets up a sample FSON network. 
	 * There are 8 layers and an output layer. Layers 1,3, and 5 are all 
	 * convoltional layers. Layers 2 and 4 are maxpool layers. Layers 6 and 7
	 * are locally connected layers, and layer 8 is a fully connected layer. 
	 * The weights of all the filters are 0.5, and all biases are 0.
	 * Please note that a save file is not initialized by this function.
	 */
	public static FSONNetwork sampleNetwork() {
		
		FSONNetwork sn = new FSONNetwork();

		// Declare and initialize the first layer
		Layer l1 = new Layer(76, 76, 3, 5, 5, 3, 32, 1, 0, LayerType.CONV);

		// Setup the layer. This creates and initializes the filters and biases, all filter weights with a value of 0.5, all biases with a value of 0.
		l1.initLayer();

		// Create the second layer.
		Layer l2 = new Layer(72, 72, 32, 3, 3, 32, 39200, 2, 0, LayerType.MAXPOOL);
		l2.initLayer();

		// Create and initialize the third layer.
		Layer l3 = new Layer(35, 35, 32, 5, 5, 32, 16, 1, 0, LayerType.CONV);
		l3.initLayer();

		// Create the fourth layer.
		Layer l4 = new Layer(31, 31, 16, 5, 5, 16, 3136, 2, 0, LayerType.MAXPOOL);
		l4.initLayer();

		// Create and initialize the fifth layer.
		Layer l5 = new Layer(14, 14, 16, 3, 3, 16, 16, 2, 0, LayerType.CONV);
		l5.initLayer();

		// Create and initialize the sixth layer.
		// This one is a locally connected layer.
		Layer l6 = new Layer(6, 6, 16, 3, 3, 1, 400, 1, 0, LayerType.LOCAL);
		l6.initLayer();

		// Create and initialize the seventh layer.
		// This one is also a locally connected layer.
		Layer l7 = new Layer(5, 5, 16, 5, 5, 16, 2048, 1, 0, LayerType.FULLY);
		l7.initLayer();

		// Create and initialize the eighth layer.
		// This one is a fully connected layer.
		Layer l8 = new Layer(2048, 1, 1, 2048, 1, 1, 2016, 1, 0, LayerType.FULLY);
		l8.initLayer();

		// This is the last "layer": this will hold the output of the network
		sn.out = new Cell[2016];

		// Initialize the cells because java won't do it for you
		for (int i = 0; i < 2016; i++) {
			sn.out[i] = new Cell();
		}

		// Initialize the list of layers
		sn.layers = new LinkedList<Layer>();

		// Add each layer at the appropriate place in the list.
		sn.layers.add(0, l1);
		sn.layers.add(1, l2);
		sn.layers.add(2, l3);
		sn.layers.add(3, l4);
		sn.layers.add(4, l5);
		sn.layers.add(5, l6);
		sn.layers.add(6, l7);
		sn.layers.add(7, l8);

		return sn;
	}

	/**
	 * This function computes the partial derivative of the total error with
	 * respect to a given *weight* within a network. This is a recursive
	 * function that operates with the help of the other function with this name
	 * and the call signature:
	 * 
	 * computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, CellCoord outcell, double[] expected)
	 * 
	 * and
	 * 
	 * computeSoftmaxError()
	 * 
	 * (both defined in this class file- IE FSONNetwork.java).
	 * 
	 * @param layers
	 *            The layers that make up this network.
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param layerIndex
	 *            The index within "layers" of the layer in which the filter
	 *            which contains the weight (that we are exploring currently)
	 *            resides.
	 * @param filterIndex
	 *            The index within the list "filters" of the filter that
	 *            contains the weight (that we are exploring currently) resides.
	 * @param depth
	 *            The depth coordinate (addressed in the order
	 *            [depth][row][column]) of the weight (that we are exploring)
	 *            within the filter.
	 * @param row
	 *            The row coordinate (addressed in the order
	 *            [depth][row][column]) of the weight (that we are exploring)
	 *            within the filter.
	 * @param column
	 *            The column coordinate (addressed in the order
	 *            [depth][row][column]) of the weight (that we are exploring)
	 *            within the filter.
	 * @param expected
	 *            The array of cells that represent the expected values of
	 *            "out".
	 * @return The calculated partial derivative of the total error with respect
	 *         to a given weight.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell, via the
	 *             compute function used on the line marked
	 *             "//Find the net value of the output cell". See
	 *             Layer::activationFunction() for more details.
	 *             
	 *TODO  Input verification? Perhaps make sure not on maxpool layer?
	 */
	public static double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, int filterIndex,
			int depth, int row, int column, double[] expected) throws Exception {
		// If there is already a value stored for this partial derivative...
		if (!Double.isNaN(layers.get(layerIndex).filters.get(filterIndex).gradientValues[depth][row][column])) {
			// ...then just use that
			//TODO: Take this out? (Will this ever be reached?)
			return layers.get(layerIndex).filters.get(filterIndex).gradientValues[depth][row][column];
		} else {
			// This will hold the sum of all the calculated partial derivatives
			// of all the connections that use this filter
			// (with respect to this weight)
			// IE: Sum( dNet(i) /dw)
			// where i = the net value of each connection's outcell
			double sum = 0;

			// For each connection for this filter (and thus uses this weight in it's calculations)...
			for (int i = 0; i < layers.get(layerIndex).filters.get(filterIndex).connections.size(); i++) {
				// Calculate dnet/dw:

				// First, grab the FilterConnection associated with this filter that we are dealing with right now.
				FilterConnection thisConnection = layers.get(layerIndex).filters.get(filterIndex).connections.get(i);

				// Find the value of the cell associated with this weight:
				// This is the CellCoordinate of the first cell in this layer used in this calculation
				CellCoord startCell = thisConnection.inStart;

				// This is to compensate for odd shaped filters and/or layers;
				// if the filter and/or layer lacks a dimension, they will be given as -1,
				// so we will count those as 0 to avoid altering where we are going to look for the cell.
				// TODO: Is this nessecary? Take this out?
				int rowTrue;
				if (row != -1) {
					rowTrue = row;
				} else {
					rowTrue = 0;
				}

				int columnTrue;
				if (column != -1) {
					columnTrue = column;
				} else {
					columnTrue = 0;
				}

				// Find the net of the cell associated with this weight
				// (the cell's value multiplied by this weight when calculating the net value for this connection).
				// Since this is the only thing multiplied by the weight when calculating, 
				// that means that this value is *also* the derivative of the net of this output cell with respect
				// to this weight(IE dnet/dw).
				// TODO: Store this instead of calculate it?
				Double dnetdw = layers.get(layerIndex).cells[startCell.depth + depth][startCell.row
						+ rowTrue][startCell.column + columnTrue].value;

				// If this is the last layer (IE the layer before "out"):
				if ((layerIndex + 1) == layers.size()) {
					sum += (dnetdw * computeSoftmaxError(out, filterIndex, expected));
				} else {

					// Since the only time we are looking for a parial derivative with respect to
					// a *weight* is when we are looking to increment that paticular weight,
					// the layer in which that weight resides *cannot* be a maxpool layer.
					//TODO: Should this be moved to earlier in the function?
					//TODO: Document this in header.
					if (layers.get(layerIndex).type == LayerType.MAXPOOL){
						throw new Exception("Trying to find a partial derivative with respect to a weight in a maxpool layer. This should never happen!");
					}	

					if (!Double.isNaN(layers.get(layerIndex
							+ 1).cells[thisConnection.out.depth][thisConnection.out.row][thisConnection.out.column].derivative)) {
						// If there is a derivative value already stored in the "out" cell for this connection,
						// use that.
						sum += (dnetdw * layers.get(layerIndex
								+ 1).cells[thisConnection.out.depth][thisConnection.out.row][thisConnection.out.column].derivative);
					} else {
						// Continue to recursively calculate the derivative
						sum += (dnetdw * computePartialDerivative(layers, out, (layerIndex + 1),
								thisConnection.out, expected));
					}

				}

			}

			//Store the newly calculated partial derivative with respect to this weight.
			layers.get(layerIndex).filters.get(filterIndex).gradientValues[depth][row][column] = sum;
			return sum;
		}

	}

	/**
	 * This function computes the partial derivative of the total error with
	 * respect to a given *bias* within a network. This is a recursive function
	 * that operates with the help of the other function with this name and the
	 * call signature:
	 * 
	 * computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, CellCoord outcell, double[] expected)
	 * 
	 * and
	 * 
	 * computeSoftmaxError()
	 * 
	 * (both defined in this class file- IE FSONNetwork.java).
	 * 
	 * @param layers
	 *            The layers that make up this network.
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param layerIndex
	 *            The index within "layers" of the layer in which the bias that
	 *            we are exploring currently resides.
	 * @param biasIndex
	 *            The index within the list "biases" of the bias that we are
	 *            exploring currently.
	 * @param expected
	 *            The array of cells that represent the expected values of
	 *            "out".
	 * @return The calculated partial derivative of the total error with respect
	 *         to a given bias.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell. See
	 *             Layer::activationFunction() for more details.
	 * 
	 *             TODO: error checking to make sure we're not on a maxpool
	 *             layer, math
	 */
	public static double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, int biasIndex,
			double[] expected) throws Exception {
		// If there is already a value stored for this partial derivative...
		// TODO: Take this out? (Will this ever be reached?)
		if (!Double.isNaN(layers.get(layerIndex).biases.get(biasIndex).derivative)) {
			// ...then just use that
			return layers.get(layerIndex).biases.get(biasIndex).derivative;
		} else {

			// This will hold the sum of all the calculated partial derivatives
			// of all the connections that use this bias
			// (with respect to this bias)
			// IE: Sum( dNet(i) / dbias)
			// where i = the net value of each connection's outcell
			double sum = 0;

			// For each filter in this layer
			for (int i = 0; i < layers.get(layerIndex).filters.size(); i++) {
				Filter currentFilter = layers.get(layerIndex).filters.get(i);

				// For each connection for this filter
				for (int j = 0; j < currentFilter.connections.size(); j++) {
					FilterConnection currentConnection = currentFilter.connections.get(j);

					// If this connection uses the bias we are looking at...
					if (currentConnection.biasIndex == biasIndex) {

						// If this is the last layer (IE the layer before "out"):
						if ((layerIndex + 1) == layers.size()) {

							// Note that since 1 is the only thing multiplied by the bias when calculating,
							// that means that 1 is the derivative of the net of this output cell 
							// with respect to this bias(IE dnet/dbias).
							// So we can ignore dnet/dbias in our calculations.
							sum += computeSoftmaxError(out, i, expected);

						} else {
							
							// Since the only time we are looking for a parial derivative with respect to
							// a *bias* is when we are looking to increment that paticular bias,
							// the layer in which that bias resides *cannot* be a maxpool layer.
							//TODO: Move this to earlier in the function?
							//TODO: Document this in header.
							if (layers.get(layerIndex).type == LayerType.MAXPOOL){
								throw new Exception("Trying to find a partial derivative with respect to a bias in a maxpool layer. This should never happen!");
							}
							
							// If this is NOT the last layer,
							// but there is a derivative value already stored in the "out" cell for this connection...
							if (!Double.isNaN(layers.get(layerIndex+ 1).cells[currentConnection.out.depth][currentConnection.out.row][currentConnection.out.column].derivative)) {

								// ... use that stored derivative value.

								// Since 1 is the only thing multiplied by the bias when calculating,
								// that means that 1 is the derivative of the net of this output cell 
								// with respect to this bias(IE dnet/dbias).
								// So we can ignore dnet/dbias in our calculations.
								sum += layers.get(layerIndex + 1).cells[currentConnection.out.depth][currentConnection.out.row][currentConnection.out.column].derivative;
							} else {

								// Continue to recursively calculate the derivative
								
								// Note that since 1 is the only thing multiplied by the bias when calculating,
								// that means that 1 is the derivative of the net of this output cell 
								// with respect to this bias(IE dnet/dbias).
								// So we can ignore dnet/dbias in our calculations.
								sum += computePartialDerivative(layers, out, (layerIndex + 1), currentConnection.out, expected);
							}

						}
					}
				}
			}
			
			//Store the newly calculated partial derivative with respect to this bias.
			layers.get(layerIndex).biases.get(biasIndex).derivative = sum;
			return sum;
		}
	}

	/**
	 * This is a helper function to the functions with the call signature:
	 * 
	 * computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int
	 * layerIndex, int filterIndex, int depth, int row, int column, double[]
	 * expected)
	 * 
	 * and
	 * 
	 * computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int
	 * layerIndex, int biasIndex, double[] expected)
	 * 
	 * (defined in this class file- IE FSONNetwork.java).
	 * 
	 * This function handles expanding the necessary derivatives for each
	 * calculation involving a cell that is not in the last layer of the network
	 * (IE the layer before "out[]". It calculates the partial derivative of the
	 * total error with respect to the net activation of a cell (that is not in
	 * the last layer). IE, it calculates de/dnet for a given cell, denoted by
	 * the CellCoord "outcell" within the layer given by the index "layerIndex".
	 * 
	 * @param layers
	 *            The layers that make up this network.
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param layerIndex
	 *            The index within "layers" of the layer in which the cell of
	 *            interest resides.
	 * @param outcell
	 *            The coordinates of the cell of interest within the layer
	 *            denoted by "layerIndex".
	 * @param expected
	 *            The array of cells that represent the expected values of
	 *            "out".
	 * @return The partial derivative of the total error with respect to the net
	 *         activation of a cell.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell, via the
	 *             parent function of this name. See:
	 * 
	 *             FSONNetwork::computePartialDerivative(LinkedList
	 *             <Layer> layers, Cell[] out, int layerIndex, int filterIndex,
	 *             int depth, int row, int column, double[] expected)
	 * 
	 *             ...for more details.
	 */
	private static Double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex,
			CellCoord outcell, double[] expected) throws Exception {

		// If there is already a value stored for this partial derivative...
		if (!Double.isNaN(layers.get(layerIndex).cells[outcell.depth][outcell.row][outcell.column].derivative)) {
			
			// ...then just use that
			return layers.get(layerIndex).cells[outcell.depth][outcell.row][outcell.column].derivative;
			//TODO: Remove this? (Will this ever be reached?)
			
		} else {
			
			// This is the calculated derivative of the out value with respect to net for this cell.
			// IE dout/dnet
			double doutdnet = layers.get(layerIndex).cells[outcell.depth][outcell.row][outcell.column].value * (1.0 - layers.get(layerIndex).cells[outcell.depth][outcell.row][outcell.column].value);
			
			// This will hold the sum of all relevant partial derivatives;
			// IE sum(dtotalerror/dout_i)
			// where i is all the weights that are applied to this cell,
			// and total error is the total error of the whole network
			// with respect to the given expected values in double[] expected.
			double dEdout = 0;

			// Find all FilterConnections that use this cell for input:

			// For all filters in this layer...
			for (int i = 0; i < layers.get(layerIndex).K; i++) {
				
				// Grab all the connections for this filter
				LinkedList<FilterConnection> currentFilterConnections = layers.get(layerIndex).filters
						.get(i).connections;
				
				for (int j = 0; j < currentFilterConnections.size(); j++) {
					
					// TODO: Account for -1s?
					// If this filter is applied to the cell we are looking
					// at...
					if ((currentFilterConnections.get(j).inStart.depth <= outcell.depth)
							&& (currentFilterConnections.get(j).inStart.row <= outcell.row)
							&& (currentFilterConnections.get(j).inStart.column <= outcell.column)
							&& ((currentFilterConnections.get(j).inStart.depth
									+ layers.get(layerIndex).Fdepth) > outcell.depth)
							&& ((currentFilterConnections.get(j).inStart.row
									+ layers.get(layerIndex).Frows) > outcell.row)
							&& ((currentFilterConnections.get(j).inStart.column
									+ layers.get(layerIndex).Fcollumns) > outcell.column)) {

						// Find the derivative of the total error with respect
						// to the output cell for this connection
						
						// Get the value of the weight multiplied by this
						// cell when this connection is calculated.
						// This gives us the partial derivative of the net
						// of the next connection with respect to the 
						// current cell
						// (IE our input paramater "outcell")
						// TODO: Missleading name. Rename variable.
						double DnetNextDoutThis;

						// Grab the depth, row, and column in the filter of
						// the weight multiplied by this cell
						// when this connection is calculated
						int depth = (outcell.depth - currentFilterConnections.get(j).inStart.depth);
						int row = (outcell.row - currentFilterConnections.get(j).inStart.row);
						int column = (outcell.column - currentFilterConnections.get(j).inStart.column);

						
						
						if (layers.get(layerIndex).type != LayerType.MAXPOOL){
							if(!Double.isNaN(layers.get(layerIndex).filters.get(i).gradientValues[depth][row][column])){
						
								//If this weight has been updated during the current iteration, we need to use the original value
								DnetNextDoutThis = layers.get(layerIndex).filters.get(i).previousWeights[depth][row][column];
								
							} else {
								
								//This is still the original value of this weight for this iteration, so we don't need to use the previous value
								DnetNextDoutThis = layers.get(layerIndex).filters.get(i).weights[depth][row][column];
								
							}
							
						} else {
							// You're never going to have a maxpool layer as the
							// last layer, so we don't have to worry about that
							// here.
							// Likewise, since an activation function is never
							// applied to a maxpool output, you don't need to
							// worry about that either!
							if (layers.get(layerIndex).cells[outcell.depth][outcell.row][outcell.column].value == layers
									.get(layerIndex + 1).cells[currentFilterConnections
											.get(j).out.depth][currentFilterConnections.get(
													j).out.row][currentFilterConnections.get(j).out.column].value) {
								DnetNextDoutThis = 1;
							} else {
								DnetNextDoutThis = 0;
							}
							
						}
						
						// If this is the last layer before out...
						if (layerIndex == (layers.size() - 1)) {
							// Add the partial derivative of the total error with respect to the output cell
							// for this connection to the sum variable
							dEdout += DnetNextDoutThis * computeSoftmaxError(out, i, expected);

						} else {
							// Recursively calculate the partial derivative of the total
							// error with respect to the output cell for this
							// connection and add it to our sum variable
							dEdout += DnetNextDoutThis *computePartialDerivative(layers, out, layerIndex + 1,
									currentFilterConnections.get(j).out, expected);
						
						}

					}
				}
			}

			double dEdnet = doutdnet* dEdout;
			// Record the newly calculated partial derivative
			layers.get(layerIndex).cells[outcell.depth][outcell.row][outcell.column].derivative = dEdnet;
			return dEdnet;
			
		}
		
	}

	/**
	 * This is a helper function for:
	 * 
	 * computePartialDerivative(LinkedList <Layer> layers, Cell[] out, int
	 * layerIndex, int filterIndex, int depth, int row, int column, double[]
	 * expected).
	 * 
	 * (defined within this class, IE FSONNetwork.java)
	 * 
	 * It handles calculating the error of a cell in the out[] array of the
	 * network.
	 * 
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param index
	 *            The index of the cell within "out" that we are calculating the
	 *            error for.
	 * @param expected
	 *            An array representing the expected values of the cells given
	 *            in "out".
	 * @return The calculated cross entropy error for the cell of index "index"
	 *         within "out".
	 */
	public static double computeSoftmaxError(Cell[] out, int index, double[] expected) {
		return (out[index].value - expected[index]);
	}

	/**
	 * This function is used to increment a single weight within a filter during
	 * the learning process.
	 * 
	 * @param learningRate
	 *            The learning rate of the learning process. How "far" each
	 *            weight moves per increment.
	 * @param layers
	 *            The layers that make up this network.
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param layerIndex
	 *            The index within "layers" of the layer in which the filter
	 *            which contains the weight (that we are incrementing) resides.
	 * @param filterIndex
	 *            The index within the list "filters" of the filter that
	 *            contains the weight (that we are incrementing) resides.
	 * @param depth
	 *            The depth coordinate (addressed in the order
	 *            [depth][row][column]) of the weight (that we are incrementing)
	 *            within the filter.
	 * @param row
	 *            The row coordinate (addressed in the order
	 *            [depth][row][column]) of the weight (that we are incrementing)
	 *            within the filter.
	 * @param column
	 *            The column coordinate (addressed in the order
	 *            [depth][row][column]) of the weight (that we are incrementing)
	 *            within the filter.
	 * @param expected
	 *            The array of cells that represent the expected values of
	 *            "out".
	 * @return The calculated new value to use for the weight.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell, via the
	 *             "computePartialDerivative" function(s). See
	 *             Layer::activationFunction() for more details.
	 */
	public static double stepGradient(double learningRate, LinkedList<Layer> layers, Cell[] out, int layerIndex,
			int filterIndex, int depth, int row, int column, double[] expected) throws Exception {
		double weight = layers.get(layerIndex).filters.get(filterIndex).weights[depth][row][column];
		double dEdweight = computePartialDerivative(layers, out, layerIndex, filterIndex, depth, row, column, expected);
		return (weight - (learningRate * dEdweight));
	}

	/**
	 * This function is used to increment a single bias during the learning
	 * process.
	 * 
	 * @param learningRate
	 *            The learning rate of the learning process. How "far" each bias
	 *            moves per increment.
	 * @param layers
	 *            The layers that make up this network
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param layerIndex
	 *            The index within "layers" of the layer in which the bias of
	 *            interest resides.
	 * @param biasIndex
	 *            The index within "biases" of the layer in which the bias
	 *            resides.
	 * @param expected
	 *            The array of cells that represent the expected values of
	 *            "out".
	 * @return The calculated new value to use for the bias.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell, via the
	 *             "computePartialDerivative" function(s). See
	 *             Layer::activationFunction() for more details.
	 */
	public static double stepGradient(double learningRate, LinkedList<Layer> layers, Cell[] out, int layerIndex,
			int biasIndex, double[] expected) throws Exception {
		double bias = layers.get(layerIndex).biases.get(biasIndex).value;
		double dEdbias = computePartialDerivative(layers, out, layerIndex, biasIndex, expected);
		return (bias - (learningRate * dEdbias));
	}

	/**
	 * This function is the main function from which learning occurs. It is a
	 * stochastic gradient descent model.
	 * 
	 * @param learningFactor
	 *            The learning rate of the learning process is a function of the
	 *            total cross entropy error of the network (calculated using the
	 *            function FSONNetwork:crossEntropyTotalError). This parameter
	 *            is a multiplier applied to that learning factor- IE this helps
	 *            control how "far" each bias or weight moves per increment.
	 * @param layers
	 *            The layers that make up this network
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param input
	 *            An array containing the filenames associated with the input to
	 *            use for learning in string form.
	 * @param iterations
	 *            How many times every weight and bias in the network is
	 *            incremented. One iteration means every bias and weight is
	 *            moved once, using a single example input.
	 * @param dictionary
	 *            This is a 2 dimensional array, containing the expected output
	 *            of the network for each input. The first dimension
	 *            (dictionary[i]) indicates which input the entry corresponds
	 *            to, and the second dimension (dictionary[][i]) contains the
	 *            expected values of out[] given that input.
	 * @param independent
	 *            This is a boolean indicating if the values in the out[] array
	 *            for the network are considered independent of one another. A
	 *            true value here indicates the values of out[] are independent
	 *            of one another. A false value here indicates the values of
	 *            out[] are dependent and must sum to 1. Practically speaking,
	 *            an independent network uses the regular sigmoid activation
	 *            function on the last "layer" (out[]), whereas a dependent
	 *            network uses the softmax activation function on the last
	 *            "layer" (out[]).
	 * @param saveFile
	 *            The filename, in string form, of a text file used to store the
	 *            network's progress while learning. This file can then be used
	 *            to recover a network if it interrupted while learning. This
	 *            file can be found in the root directory of this project.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell. See
	 *             Layer::activationFunction() for more details.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input. See openFileInput (declared in this class file) for
	 *             more details.
	 */
	public static void learn(double learningFactor, LinkedList<Layer> layers, Cell[] out, String[] input,
			int iterations, double[][] dictionary, boolean independent, String saveFile) throws Exception {

		// Grab the location of this class file in the filesystem
		URL location = FSONNetwork.class.getProtectionDomain().getCodeSource().getLocation();

		String urlString = location.toString();

		// Chop the end off the string,
		// resulting in the filepath of the root of this project
		String substr = urlString.substring(5, (urlString.length() - 4));

		// Add the filename to the end of the path.
		// Now we have the absolute path of a file located in the root directory of this project.
		String absPath = substr.concat(saveFile);
		
		// Find the starting error.
		System.out.println("Calculating error before learning.");
		double totalError = crossEntropyTotalError(layers, out, input, dictionary, independent);
		System.out.println("Starting learning. Error is: " + totalError);

		// Calculate the starting learning rate using the starting error and the learningFactor parameter
		double learningRate = Layer.activationFunction(totalError) * learningFactor;

		// For the requested number of iterations...
		for (int i = 0; i < iterations; i++) {

			// 1. Pick an example, feed it forward through the network:
			// 1.a) Generate a random order in which to access the input:
			ArrayList<Integer> randomList = UniqueRandomNumbers.getRandomSet(input.length);

			// 1.b) Use "randomList" to access each input entry in a random order
			for (int r = 0; r < input.length; r++) {
				int s = randomList.get(r);

				// If there is an input to be learned at this index...
				if ((input[s] != null) && (!input[s].equals(""))){
					
					// There might be multiple inputs at this index
					// So split the input...
					String[] inputs = input[s].split(",");
					int numInputs = inputs.length;
					
					//...and access each input in turn
					for (int n = 0; n<numInputs; n++){
						
						// Open the next input file denoted by the string stored
						// in the input array, in a random order, and feed that input
						// into the first layer.

						// If the first layer only has a depth of 1, that means the
						// input is supposed to be black and white, so use the
						// appropriate function to open it
						if (layers.getFirst().cells.length == 1) {
							openFileInputBW(layers, inputs[n]);
						} else { // If the first layer has more than a single depth,
							// that means it is expecting an image with multiple
							// channels, so use the appropriate function to open
							// it
							openFileInput(layers, inputs[n]);
						}

						// 1.b.ii) Feed the input through the rest of the network
						feedForward(layers, out, false);

						// If the output cells are independent of one another, use the sigmoid activation function
						if (independent) {
							for (int w = 0; w < out.length; w++) {
								out[w].value = Layer.activationFunction(out[w].value);
							}
						} else {// The output cells are dependent, and so we must use the softmax activation function
							Layer.softmax(out);
						}

						// 2. Increment all weights:

						// 2.a) Increment all weights for all the layers, working backward.
						for (int j = (layers.size() - 1); j >= 0; j--) {
						//	System.out.println("----------------------------------------------");
						//	System.out.println("Processing layer: " + j);
						//	System.out.println("----------------------------------------------");
							Layer currentLayer = layers.get(j);
							
							if (currentLayer.type!=LayerType.MAXPOOL){
								// 2.a.i) Increment all weights for all the filters for this layer ("currentLayer")
								for (int f = 0; f < currentLayer.filters.size(); f++) {
								//	System.out.println("Incrementing filter: " + f);
									Filter currentFilter = currentLayer.filters.get(f);

									// 2.a.i.1) Increment each weight within this filter ("currentFilter")
									for (int x = 0; x < currentLayer.Fdepth; x++) {
										for (int y = 0; y < currentLayer.Frows; y++) {
											for (int z = 0; z < currentLayer.Fcollumns; z++) {
												// Note that "dictionary[s]" is used because
												// the sth entry in the dictionary is the expected output for this input
												// (the "n"th entry in the input at input[s]).
												currentFilter.previousWeights[x][y][z] = currentFilter.weights[x][y][z];
												currentFilter.weights[x][y][z] = stepGradient(learningRate, layers, out, j, f, x, y, z, dictionary[s]);
											}
										}
									}
								}

								// 2.b) Increment all the biases for this layer:
								for (int b = 0; b < currentLayer.biases.size(); b++) {
									// Note that "dictionary[s]" is used because the sth
									// entry in the dictionary is the expected output for
									// this input (the "n"th entry in the input at input[s])
								//	System.out.println("Incrementing bias: "+ b);
									currentLayer.biases.get(b).previousValue = currentLayer.biases.get(b).value;
									currentLayer.biases.get(b).value = stepGradient(learningRate, layers, out, j, b, dictionary[s]);
								}
							}

						}

						
					//	System.out.println("----------------------------------------------");
					//	System.out.println("Recording and resetting layers");
					//	System.out.println("----------------------------------------------");
						// Open a file, using the path created above, to store our progress while learning.
						// This file can then be used to recover a network if we are interrupted while learning.
						PrintWriter fw = new PrintWriter(absPath);
						
						// 3. Reset stored gradients and write the new weights to the file
						// 3.a) Reset all stored gradients for all layers
						for (int j = 0; j < layers.size(); j++) {
						//	System.out.println("----------------------------------------------");
						//	System.out.println("Recording layer: "+ j);
						//	System.out.println("----------------------------------------------");
							// Record this layer
							fw.write("<layer>\n");
							
							Layer currentLayer = layers.get(j);

							// Record the paramaters for this layer
							fw.write(currentLayer.collumns + "," + currentLayer.rows+ "," + currentLayer.depth + "," + currentLayer.Fcollumns + "," + currentLayer.Frows + "," + currentLayer.Fdepth + "," + currentLayer.K + ","+ currentLayer.step+ ","+ currentLayer.pad+ "," + currentLayer.type+"\n");
							fw.flush();

							// Record the cells of this layer
							fw.write("<cells>\n");
							
						//	System.out.println("Recording cells");
							// 3.c) Reset all stored gradients for all the cells in this layer, recording their value and derivative first
							for (int x = 0; x < currentLayer.depth; x++) {
								for (int y = 0; y < currentLayer.rows; y++) {
									for (int z = 0; z < currentLayer.collumns; z++) {
										
										fw.write(currentLayer.cells[x][y][z].value + "," + currentLayer.cells[x][y][z].derivative+"\n");
										currentLayer.cells[x][y][z].derivative = Double.NaN;
									}
								}
							}
							fw.flush();

							if(currentLayer.type != LayerType.MAXPOOL){

								// 3.a) Reset all stored gradients for all the filters for this layer ("currentLayer")
								for (int f = 0; f < currentLayer.filters.size(); f++) {
									Filter currentFilter = currentLayer.filters.get(f);
								//	System.out.println("Recording filter: "+f );
									// Record this filter
									fw.write("<filter>\n");

									// 3.a.i) Reset all stored gradients for each weight within this filter ("currentFilter"), recording the filter weights and gradients first
									for (int x = 0; x < currentLayer.Fdepth; x++) {
										for (int y = 0; y < currentLayer.Frows; y++) {
											for (int z = 0; z < currentLayer.Fcollumns; z++) {
												fw.write(currentFilter.weights[x][y][z] + ","+ currentFilter.gradientValues[x][y][z]+"\n");
												currentFilter.gradientValues[x][y][z] = Double.NaN;
											}
										}
									}

									// Record the connections of this filter
									for (int x = 0; x < currentFilter.connections.size(); x++){
										FilterConnection currentConnection = currentFilter.connections.get(x);
										fw.write("<connection>\n");
										fw.write(currentConnection.biasIndex+","+ currentConnection.inStart.depth +","+ currentConnection.inStart.row +","+ currentConnection.inStart.column +"," + currentConnection.out.depth +","+ currentConnection.out.row +","+currentConnection.out.column+"\n");
										fw.flush();
									}

									// Indicate the end of this filter's data
									fw.write("</filter>\n");
									fw.flush();
								}

								// 3.b) Reset all stored gradients for all the biases for this layer, recording their values and derivatives first
								for (int b = 0; b < currentLayer.biases.size(); b++) {
								//	System.out.println("Recording bias: "+b );
									fw.write("<bias>\n");
									fw.write(currentLayer.biases.get(b).derivative + "," + currentLayer.biases.get(b).value+"\n");
									currentLayer.biases.get(b).derivative = Double.NaN;
								}

								// If we have recorded a bias, indicate the end of the list of biases
								if (currentLayer.biases.size() >0){
									fw.write("</biases>\n");
								}
							}

							// Indicate the end of this layer's data
							fw.write("</layer>\n");
							fw.flush();
							fw.close();
						}
					}
					
					// Recalculate the total error
					totalError = crossEntropyTotalError(layers, out, input, dictionary, independent);
					// Recalculate the learning rate
					learningRate = Layer.activationFunction(totalError) * learningFactor;
				}
				
			}
			
		//	System.out.println("----------------------------------------------");
		//	System.out.println("Recalculating error");
		//	System.out.println("----------------------------------------------");
			System.out.println("Iteration " + i + " complete. Error is now: " + totalError);

			
		}
	}

	/**
	 * This function opens a single file (indicated by the filename passed in as
	 * a string), and feeds that file as input into the first layer of the
	 * network. Input files are expected to be in color and in .jpg form.
	 * 
	 * @param layers
	 *            The layers that make up this network
	 * @param filename
	 *            The filename of the file to be opened/used, in string form.
	 *            Note that the file must be in the root directory of this
	 *            project.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell. See
	 *             Layer::activationFunction() for more details.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input.
	 */
	public static void openFileInput(LinkedList<Layer> layers, String filename) throws Exception {
		
		// Grab the location of this class file in the filesystem
		URL location = FSONNetwork.class.getProtectionDomain().getCodeSource().getLocation();

		String urlString = location.toString();

		// Chop the end off the string,
		// resulting in the filepath of the root of this project
		String substr = urlString.substring(5, (urlString.length() - 4));

		// Add the filename to the end of the path.
		// Now we have the absolute path of a file located in the root directory
		// of this project
		String absPath = substr.concat(filename);
		
//		System.out.println("Reading: "+absPath);

		// This is necessary to use any of the OpenCV functions
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// Actually read the file into a Mat object
		Mat img = Highgui.imread(absPath, 1);

		// "imread" fails silently,
		// so be sure to check the file to make sure it was read successfully.
		if (img.cols() == 0) {
			throw new IOException("Error reading file: "+ filename);
		}

		// This is necessary to use any of the OpenCV functions
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// Resize the image to the size needed.
		Mat resizedImage = new Mat();
		Size sz = new Size(layers.get(0).collumns, layers.get(0).rows);
		Imgproc.resize(img, resizedImage, sz);

		// Split the image into desired number of color channels
		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here in the order RGB
		Core.split(resizedImage, channels);

		// For each channel...
		for (int c = 0; c < channels.size(); c++) {

			// Feed the individual pixel values into a temporary array...
			channels.get(c).convertTo(channels.get(c), CvType.CV_64FC3);
			int size = (int) (channels.get(c).total() * channels.get(c).channels());
			double[] temp = new double[size];
			channels.get(c).get(0, 0, temp);

			// ...and then into the cells of the first layer of the network.
			for (int d = 0; d < layers.get(0).rows; d++) {
				for (int e = 0; e < layers.get(0).collumns; e++) {
					
					// A single channel Mat has pixels of a value between 0 and 255
					// inclusive, where 0 is min value and 255 is max value.
					// Since the network is expecting a value between 0 and 1, some
					// formatting of the data is required.
					double value = temp[(d * layers.get(0).rows) + e];
					value = (value-127.5)/25.5;
					layers.get(0).cells[c][d][e].value = Layer.activationFunction(value);
					
				}

			}

		}
	}

	/**
	 * This function opens a single file (indicated by the filename passed in as
	 * a string), and feeds that file as input into the first layer of the
	 * network. Input files are expected to be in black and white and .jpg form.
	 * 
	 * @param layers
	 *            The layers that make up this network
	 * @param filename
	 *            The filename of the file to be opened/used, in string form.
	 *            Note that the file must be in the root directory of this 
	 *            project.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell. See
	 *             Layer::activationFunction() for more details.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input.
	 * 
	 */
	public static void openFileInputBW(LinkedList<Layer> layers, String filename) throws Exception {

		// Grab the location of this class file in the filesystem
		URL location = FSONNetwork.class.getProtectionDomain().getCodeSource().getLocation();

		String urlString = location.toString();

		// Chop the end off the string,
		// resulting in the filepath of the root of this project
		String substr = urlString.substring(5, (urlString.length() - 4));

		// Add the filename to the end of the path.
		// Now we have the absolute path of a file located in the root directory of this project
		String absPath = substr.concat(filename);

//		System.out.println("Reading: "+absPath);

		// This is necessary to use any of the OpenCV functions
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// Actually read the file into a Mat object
		Mat img = Highgui.imread(absPath, 0);

		// "imread" fails silently,
		// so be sure to check the file to make sure it was read successfully.
		if (img.cols() == 0) {
			throw new IOException("Error reading file");
		}

		// Next, resize the image to the size needed.
		Mat resizedImage = new Mat();
		Size sz = new Size(layers.get(0).collumns, layers.get(0).rows);
		Imgproc.resize(img, resizedImage, sz);

		// ...and then feed the values of the pixels into the cells of the first layer of the network.
		for (int d = 0; d < layers.get(0).rows; d++) {
			for (int e = 0; e < layers.get(0).collumns; e++) {
				
				// A black and white Mat has pixels of a value between 0 and 255
				// inclusive, where 0 is black and 255 is white.
				// Since the network is expecting a value between 0 and 1, some
				// formatting of the data is required.
				double value =resizedImage.get(d,e)[0];
				value = (value-127.5)/25.5;
				layers.get(0).cells[0][d][e].value = Layer.activationFunction(value);
				
			}

		}

	}

	/**
	 * This function carries out a single forward pass through the network.
	 * Please note that the desired input must already be loaded into the first
	 * layer before calling this function (you can use "openFileInput" defined
	 * in this class for this purpose).
	 * 
	 * @param layers
	 *            The layers that make up this network
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param store
	 *            An indication if this network should be set up or not. A
	 *            "true" value here means that connections should be recorded
	 *            because this is the first pass through the network and the
	 *            structure needs to be recorded. A false value here means this
	 *            is network is already set up and does not need to record
	 *            connections (no connections will be recorded in this call).
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell. See
	 *             Layer::activationFunction() for more details.
	 */
	public static void feedForward(LinkedList<Layer> layers, Cell[] out, boolean store) throws Exception {

		//For all layers in this network but the last one before out...
		for (int i = 0; i < (layers.size() - 1); i++) {
			Layer currentLayer = layers.get(i);
			Layer nextLayer = layers.get(i + 1); //This is needed because the next layer contains the output of the first

			switch (currentLayer.type) {
			case CONV:
				currentLayer.convolution(currentLayer.cells, currentLayer.filters, nextLayer.cells, currentLayer.step,
						currentLayer.pad, currentLayer.biases, store, true);
				break;
			case FULLY:
				currentLayer.full(currentLayer.cells, currentLayer.filters, nextLayer.cells[0][0], currentLayer.step,
						currentLayer.pad, currentLayer.biases, store, true);
				break;
			case LOCAL:
				currentLayer.local(currentLayer.cells, currentLayer.filters, nextLayer.cells, currentLayer.step,
						currentLayer.pad, currentLayer.biases, store, true);
				break;
			case MAXPOOL:
				currentLayer.pool(currentLayer.cells, currentLayer.filters, nextLayer.cells, currentLayer.step,
						currentLayer.Fcollumns, store);
				break;
			default:
				// TODO:Throw exception/error here
				break;

			}
		}

		// Calculate the last layer's output into "out".
		// This will always be a fully connected layer.
		Layer lastLayer = layers.getLast();
		lastLayer.full(lastLayer.cells, lastLayer.filters, out, lastLayer.step, lastLayer.pad, lastLayer.biases, store,
				false);

	}

	/**
	 * This function calculates and returns the total cross entropy error over
	 * all inputs into the network. For a good explanation of cross entropy
	 * loss, I recommend:
	 * http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross
	 * -entropy_cost_function
	 * 
	 * @param layers
	 *            The layers that make up this network
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param input
	 *            An array containing the filenames associated with the input to
	 *            use for learning in string form.
	 * @param dictionary
	 *            This is a 2 dimensional array, containing the expected output
	 *            of the network for each input. The first dimension
	 *            (dictionary[i]) indicates which input the entry corresponds
	 *            to, and the second dimension (dictionary[][i]) contains the
	 *            expected values of out[] given that input.
	 * @param independent
	 *            This is a boolean indicating if the values in the out[] array
	 *            for the network are considered independent of one another. A
	 *            true value here indicates the values of out[] are independent
	 *            of one another. A false value here indicates the values of
	 *            out[] are dependent and must sum to 1. Practically speaking,
	 *            an independent network uses the regular sigmoid activation
	 *            function on the last "layer" (out[]), whereas a dependent
	 *            network uses the softmax activation function on the last
	 *            "layer" (out[]).
	 * @return The total cross entropy error over all inputs into the network
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell. See
	 *             Layer::activationFunction() for more details.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input.
	 *             
	 *             TODO: Could maybe use better documentation?
	 */
	public static double crossEntropyTotalError(LinkedList<Layer> layers, Cell[] out, String[] input,
			double[][] dictionary, boolean independent) throws Exception {
		
		double sum = 0;

		// For each example in the training data...
		for (int i = 0; i < input.length; i++) {

			//If there is an input to be learned at this index...
			if ((input[i] != null) && (!input[i].equals(""))){
				
				String[] inputs = input[i].split(",");
				int numInputs = inputs.length;
				for (int n = 0; n<numInputs; n++){
					// Open that example and feed it through the network:
					// If the first layer only has a depth of 1, that means the input is
					// supposed to be black and white, so use the appropriate function
					// to open it
					if (layers.getFirst().cells.length == 1) {
						openFileInputBW(layers, inputs[n]);
					} else { // If the first layer has more than a single depth, that
								// means it is expecting an image with multiple
								// channels, so use the appropriate function to open it
						openFileInput(layers, inputs[n]);
					}
					
					// Feed the input through the network (conduct a forward pass)
					feedForward(layers, out, false);

					// If the output cells are independent of one another, use the sigmoid activation function
					if (independent) {
						for (int w = 0; w < out.length; w++) {
							out[w].value = Layer.activationFunction(out[w].value);
						}
					} else {// The output cells are dependent, and so we must use the softmax activation function
						Layer.softmax(out);
					}

					// For each cell in out...
					for (int k = 0; k < out.length; k++) {
						// Add y*ln(x) + (1-y)*ln(1-x) to the sum,
						// where y is the expected value for this cell
						// and x is the actual value for this cell
							double log;
							if (out[k].value == 0){
								log = Double.NEGATIVE_INFINITY;
							}else{
								log = Math.log(out[k].value);
							}
							
							double oneMinusLog;
							
							if (out[k].value == 1){
								oneMinusLog = Double.NEGATIVE_INFINITY;
				
							} else{
								oneMinusLog = Math.log(1 - out[k].value);
							}
							
							sum += (dictionary[i][k] * log)
									+ ((1 - dictionary[i][k]) * oneMinusLog);

					}
				}
			}

		}
		// Multiply sum by -1/n, where n is the number of examples in the training data
		double error = (sum * (0 - (1)));

		return error;
	}
	
	/**
	 * This function locates a name within an alphabetically sorted array of names
	 * and returns the index at which that name was found. 
	 * It uses a recursive "divide and conquor" algorythm.
	 *  
	 * @param names The master array of names to search
	 * @param start The starting index of the section to search
	 * @param end The ending index of the section to search
	 * @param toFind The name to find within the section searched of the array
	 * @return The index of the location of the name within the "names" array, or -1 if that name is not found 
	 */
	public static int findName(String[] names, int start, int end, String toFind) {
        
        if (start < end) {
            int middleIndex = ((end - start) / 2) + start; 
            if ((toFind.compareTo(names[middleIndex])) < 0){
				return findName(names, start, middleIndex, toFind);

			} else if ((toFind.compareTo(names[middleIndex])) > 0) {
				return findName(names, middleIndex + 1, end, toFind);

			} else {
				return middleIndex;
			}
		} else if (toFind.compareTo(names[end]) == 0) {
			return end;
		} else {
			return -1;
		}

	}
	
	/**
	 * This function sets up the input for learning using data provided by 
	 * Labled Faces in the Wild (http://vis-www.cs.umass.edu/lfw/).
	 * It creates an array the length of the total number of names in the 
	 * LFW data set, with a comma seperated list of filenames of known images of 
	 * each person to use for learning.
	 * The master list of names is read from the file lfw_all_names.txt in the root
	 * directory of this project. 
	 * 
	 * @throws Exception An exception is thrown if an error during file IO is encountered
	 */
	public void learnLFW() throws Exception{
		
		String[] learnInput = readInLFWData();
		
		// This is the dictionary used to tell the learning functions what the ideal
		// output for a picture of that person would look like
		double[][] dictionary = new double[5749][5749];
		
		// Since each entry in the array corresponds to a different person
		// that means that the ideal output would be an array of all 0s
		// except for the entry at that person's index.
		// Since java initializes arrays to 0 by default, that means that
		// all we need to do is set the "1"
		for(int k = 0; k <dictionary.length; k++){
			dictionary[k][k] =1;
		}
		
		// Use the learning function to learn using our newly processed input and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		learn(1, this.layers, this.out, learnInput, 1, dictionary, false, "testlearnlfw.txt");
		
	}

	public static FSONNetwork simpleNetwork() {
		
		FSONNetwork sn = new FSONNetwork();

		// Declare and initialize the first layer
		Layer l1 = new Layer(12, 12, 3, 4, 4, 3, 3, 1, 0, LayerType.CONV);

		// Setup the layer. This creates and initializes the filters and biases, all filter weights with a value of 0.5, all biases with a value of 0.
		l1.initLayer();

		// Create the second layer.
		Layer l2 = new Layer(10, 10, 3, 2, 2, 1, 75, 2, 0, LayerType.MAXPOOL);
		l2.initLayer();

		// Create and initialize the third layer.
		Layer l3 = new Layer(5, 5, 3, 3, 3, 3, 9, 1, 0, LayerType.LOCAL);
		l3.initLayer();

		// Create and initialize the fourth layer.
		Layer l4 = new Layer(3, 3, 1, 3, 3, 1, 5, 1, 0, LayerType.FULLY);
		l4.initLayer();

		// This is the last "layer": this will hold the output of the network
		sn.out = new Cell[5];

		// Initialize the cells because java won't do it for you
		for (int i = 0; i < 5; i++) {
			sn.out[i] = new Cell();
		}

		// Initialize the list of layers
		sn.layers = new LinkedList<Layer>();

		// Add each layer at the appropriate place in the list.
		sn.layers.add(0, l1);
		sn.layers.add(1, l2);
		sn.layers.add(2, l3);
		sn.layers.add(3, l4);

		return sn;
	}
	
	public void learnLFWSimple() throws Exception{
		
		String[] learnInput = readInLFWData();
		
		// This is the dictionary used to tell the learning functions what the ideal
		// output for a picture of that person would look like
		double[][] dictionary = new double[5749][5];
		
		// Given an index, "dictionary[x][y]", 
		// x is the index in the list of names of the person the input represents
		// and y is the array of output we would expect to see in a perfectly trained network.
		// Since this is a simplified version, this networks "out" array is only 5 cells long,
		// so we can't expect it to recognize all the faces. 
		// So I've given it 4 faces to recognize and a 5th cell to indicate the input does
		// not represent any of those four faces.
		for(int k = 0; k <dictionary.length; k++){
			switch (k) {
			case 5:
				//"Aaron_Peirsol"
				dictionary[k][0] = 1;
				dictionary[k][1] = 0;
				dictionary[k][2] = 0;
				dictionary[k][3] = 0;
				dictionary[k][4] = 0;
				break;
			case 36:
				//"Adam_Sandler"
				dictionary[k][0] = 0;
				dictionary[k][1] = 1;
				dictionary[k][2] = 0;
				dictionary[k][3] = 0;
				dictionary[k][4] = 0;
				break;
			case 118:
				//"Alec_Baldwin"
				dictionary[k][0] = 0;
				dictionary[k][1] = 0;
				dictionary[k][2] = 1;
				dictionary[k][3] = 0;
				dictionary[k][4] = 0;
				break;
			case 222:
				//"Amelia_Vega"
				dictionary[k][0] = 0;
				dictionary[k][1] = 0;
				dictionary[k][2] = 0;
				dictionary[k][3] = 1;
				dictionary[k][4] = 0;
				break;
				default:
					//A person that is not one of the above 4 people
					dictionary[k][0] = 0;
					dictionary[k][1] = 0;
					dictionary[k][2] = 0;
					dictionary[k][3] = 0;
					dictionary[k][4] = 1;
					break;
			
			}
		}
		
		// Use the learning function to learn using our newly processed input and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		learn(1, this.layers, this.out, learnInput, 100, dictionary, false, "testlearnlfwsimple.txt");
		
	}
	
	public static FSONNetwork simpleNetworkBinary() {
		
		FSONNetwork snb = new FSONNetwork();

		// Create and initialize the fourth layer.
		Layer l4 = new Layer(100, 100, 3, 100, 100, 3, 2, 1, 0, LayerType.FULLY);
		l4.initLayer();

		// This is the last "layer": this will hold the output of the network
		snb.out = new Cell[2];

		// Initialize the cells because java won't do it for you
		for (int i = 0; i < 2; i++) {
			snb.out[i] = new Cell();
		}

		// Initialize the list of layers
		snb.layers = new LinkedList<Layer>();

		// Add each layer at the appropriate place in the list.
		snb.layers.add(0, l4);

		return snb;
	}
	
	public void learnLFWSimpleBin() throws Exception{
		
		// Grab the location of this class file in the filesystem
		URL location = FSONNetwork.class.getProtectionDomain().getCodeSource().getLocation();

		String urlString = location.toString();

		// Chop the end off the string,
		// resulting in the filepath of the root of this project
		String substr = urlString.substring(5, (urlString.length() - 4));
		
		// Add the filename to the end of the path.
		// This is the file containing the names of the pairs we are going to learn from.
		String pairsPath = substr.concat("testingInput/pairsDevTrain4.txt");

		// Create a second reader to read in the training names
		BufferedReader br = new BufferedReader(new FileReader(pairsPath));
		
		// This will hold a line of text from the file
		String line = null;
		
		// Read the first line of the file containing the training pairs.
		// This will be the number of training pairs to expect.
		// (N matches and N mismatches)
		line = br.readLine();
		//Split the line up by commas
		String[] names = line.split(",");
		
		// Read the first line of the file containing the training pairs.
		// This will be the number of training pairs to expect.
		// (N matches and N mismatches)
		line = br.readLine();
		//Split the line up by commas
		String[] names2 = line.split(",");
		
		br.close();

		int numb = names.length+names2.length;

		// This array will hold the filenames of the pictures of each person, in a comma seperated list
		String[] learnInput = new String[numb];
		
		for (int i=0; i<names.length; i++){
			String filename = "lfw/"+ names[i]+ "/" +names[i]+ "_0001.jpg";
			learnInput[i] = filename;
		}
		
		for (int i = 0; i<names2.length; i++){
			String filename = "testingInput/"+ names2[i]+ ".jpg";
			learnInput[i+names.length] = filename;
		}
		
		// This is the dictionary used to tell the learning functions what the ideal
		// output for a picture of that person would look like
		double[][] dictionary = new double[learnInput.length][2];
		
		// Given an index, "dictionary[x][y]", 
		// x is the index in the list of names of the person the input represents
		// and y is the array of output we would expect to see in a perfectly trained network.
		// Since this is a simplified version, this networks "out" array is only 5 cells long,
		// so we can't expect it to recognize all the faces. 
		// So I've given it 4 faces to recognize and a 5th cell to indicate the input does
		// not represent any of those four faces.
		for(int k = 0; k <dictionary.length; k++){
			if (k <names.length){
				dictionary[k][0] = 1.0;
				dictionary[k][1] = 0.0;
			} else{
				dictionary[k][0] = 0.0;
				dictionary[k][1] = 1.0;			
			}
		}
		
		FSONNetwork.openFileInput(layers, learnInput[0]);
		
//		double test[][][]= new double[layers.getFirst().depth][layers.getFirst().rows][layers.getFirst().collumns];
//		// Depth
//		for (int l = 0; l < layers.getFirst().depth; l++) {
//			// Row
//			for (int j = 0; j < layers.getFirst().rows; j++) {
//				// Column
//				for (int k = 0; k < layers.getFirst().collumns; k++) {
//					test[l][j][k]= layers.getFirst().cells[l][j][k].value;
//				}
//			}
//		}
		FSONNetwork.feedForward(layers, out, true);
		// Use the learning function to learn using our newly processed input and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		learn(0.005, this.layers, this.out, learnInput, 10, dictionary, false, "testlearnlfw.txt");
		
	}

	private String[] readInLFWData() throws FileNotFoundException, IOException, Exception {
		// Grab the location of this class file in the filesystem
		URL location = FSONNetwork.class.getProtectionDomain().getCodeSource().getLocation();

		String urlString = location.toString();

		// Chop the end off the string,
		// resulting in the filepath of the root of this project
		String substr = urlString.substring(5, (urlString.length() - 4));

		// Add the filename to the end of the path.
		// This is the file containing *all* the names.
		String absPath = substr.concat("lfw_all_names.txt");

		// Create a reader to read the names
		BufferedReader br = new BufferedReader(new FileReader(absPath));
		
		// This will hold a line of text from the file
		String line = null;
		
		// This will hold all the names
		String[] names = new String[5749];
		
		// Read the names in
		for(int i = 0; ((i< 5749) &&((line = br.readLine()) != null)); i++) {
			String[] pair = line.split("	");
			names[i] = pair[0];	
		}
		
		// Close the reader
		br.close();
		
		// Add the filename to the end of the path.
		// This is the file containing the names of the pairs we are going to learn from.
		String pairsPath = substr.concat("testingInput/pairsDevTrain3.txt");

		// Create a second reader to read in the training names
		BufferedReader br2 = new BufferedReader(new FileReader(pairsPath));
		
		// This will hold a line of text from the file
		String line2 = null;
		
		// Read the first line of the file containing the training pairs.
		// This will be the number of training pairs to expect.
		// (N matches and N mismatches)
		line2 = br2.readLine();
		int N = Integer.parseInt(line2);
		
		// This array will hold the filenames of the pictures of each person, in a comma seperated list
		String[] learnInput = new String[5749];

		//This is used to help format the filenames
		DecimalFormat formatter = new DecimalFormat("0000");
		
		// Read in the first set of learning pairs (the matched pairs)
		// These lines will be in the format:
		// name	x	y
		//...where  x and y are the # denoting a filename of a picture of that person
		for(int i = 0; ((i< N) &&((line2 = br2.readLine()) != null)); i++)  {
			
			//Split the line up by tabs
			String[] pair = line2.split("	");
			
			// Find the index associated with the name just read in
			int index = findName(names,0,5748, pair[0]);
			
			//Construct the filenames, using the numbers read in
			String filename1 = "lfw/"+ pair[0]+ "/" +pair[0].concat("_")+formatter.format(Double.parseDouble(pair[1]))+".jpg";
			String filename2 = "lfw/"+ pair[0]+ "/" +pair[0].concat("_")+formatter.format(Double.parseDouble(pair[2]))+".jpg";
		
			// If the name is found in the list of names
			// and the entry at that index in learnInput is not empty 
			// (IE filenames of input for that person have already been recorded)
			if ((learnInput[index] != null) && (!learnInput[index].isEmpty())){
				
				// Grab the filenames at this index. They will be delemited by a comma.
				String[] inputsAtIndex = learnInput[index].split(",");
				
				// These booleans indicate if the filenames we read in from the learning pairs
				// are found within the filenames already stored at this index
				boolean found1 = false;
				boolean found2 = false;
				
				// Search the filenames stored at this index, recording if we find either 
				// filename1 (by setting found1 to true) or filename2 (by setting found2 to 
				// true
				for (int j = 0; j< inputsAtIndex.length; j++){
					if (inputsAtIndex[j].equals(filename1)){
						found1 = true;
					} else if(inputsAtIndex[j].equals(filename2)){
						found2= true;
					}
				}
				
				// If we haven't found filename1 alreaded stored at this index,
				// simply add it to the end of the string at this index with a
				// delimiting comma
				if (!found1){
					learnInput[index] = learnInput[index].concat(","+filename1);
				}
				
				// If we haven't found filename2 alreaded stored at this index,
				// simply add it to the end of the string at this index with a
				// delimiting comma
				if (!found2){
					learnInput[index] = learnInput[index].concat(","+filename2);
				}
				
			} else{ // The entry at this index doesn't have any input filenames stored
				// here yet
				
				//If this is the first input filename processed...
				if (i == 0){
					//Open it as input
					openFileInput(layers, filename1);
					//Feed it forward and set up the connections
					feedForward(layers, out, true);
				}
				
				// If the filenames aren't the same
				if (!(filename1.equals(filename2))){
					// Store them both at the entry for this index,
					// seperated by a comma
					learnInput[index] = filename1.concat(","+filename2);
				} else {
					// Just store the first filename, because they are 
					// the same.
					learnInput[index] = filename1;
				}
				
			}
		}

		// This section processes the learning pairs that represent mismatches
		// These lines will be in the format:
		// name	x	name2 y
		//...where  x and y are the # denoting a filename of a picture of the person named before it
		// (IE "x" is a filename of an image of "name", "y" is a filename of an image of "name2")
		for(int j = 0; ((j< N) &&((line2 = br2.readLine()) != null)); j++)  {
			
			// Split the line up by tabs
			String[] pair = line2.split("	");
			
			// Find the index of the name of the first person in the pair
			// in the master list of names
			int index1 = findName(names,0,5748, pair[0]);
			
			// Find the index of the name of the second person in the pair
			// in the master list of names
			int index2 = findName(names,0,5748, pair[2]);
			
			// Build the filenames for each image
			String filename1 = "lfw/"+ pair[0]+ "/" +pair[0].concat("_")+formatter.format(Double.parseDouble(pair[1]))+".jpg";
			String filename2 = "lfw/"+ pair[2]+ "/" +pair[2].concat("_")+formatter.format(Double.parseDouble(pair[3]))+".jpg";
		

			// First deal with the first image ("filename1", a picture of the person named at "index1")
			// If the name was found in the master list of names (ie the entry at this index is not null)
			// and the entry at this index already contains filenames
			if ((learnInput[index1] != null) && (!learnInput[index1].isEmpty())){
				
				// Split the string containing the list of filenames up by the delimiting commas
				String[] inputsAtIndex = learnInput[index1].split(",");
				
				// This boolean will tell us if we find filename1 in this list
				boolean found1 = false;
				
				// Search the list of filenames contained in the entry at this index for filename1
				int k =0;
				while ((k< inputsAtIndex.length)&&(!found1)){
					if (inputsAtIndex[k].equals(filename1)){
						found1 = true;
					} else {
					k++;
					}
				}
				
				// If we don't find filename1 in the list of filenames at this entry...
				if (!found1){
					//... add it to the end of the list with a delimiting comma
					learnInput[index1] = learnInput[index1].concat(","+filename1);
				}
				
			} else{ // The entry at this index is empty
				learnInput[index1] = filename1; // So store the filename here
			}
			
			// Now we deal with the second image ("filename2", a picture of the person named at "index2")
			// If the name was found in the master list of names (ie the entry at this index is not null)
			// and the entry at this index already contains filenames
			if ((learnInput[index2] != null) && (!learnInput[index2].isEmpty())){
				
				// Split the string containing the list of filenames up by the delimiting commas
				String[] inputsAtIndex = learnInput[index2].split(",");
				
				// This boolean will tell us if we find filename2 in this list
				boolean found2 = false;
				
				// Search the list of filenames contained in the entry at this index for filename2
				int k =0;
				while ((k< inputsAtIndex.length)&&(!found2)){
					if (inputsAtIndex[k].equals(filename1)){
						found2 = true;
					} else {
					k++;
					}
				}
				
				// If we don't find filename2 in this list
				if (!found2){
					
					// Add filename2 to the end of the list, seperated by a comma
					learnInput[index1] = learnInput[index2].concat(","+filename2);
				}
				
			} else{ // There aren't any filenames stored at this index yet
				learnInput[index2] = filename2; // So set this entry to filename2
			}
			
		}
		
		// Close the reader used to read in the training pairs
		br2.close();
		return learnInput;
	}
	
	public static FSONNetwork flagNetwork() {
		
		FSONNetwork fn = new FSONNetwork();

		// Declare and initialize the first layer
		Layer l1 = new Layer(40, 40, 3, 5, 5, 3, 16, 1, 0, LayerType.CONV);

		// Setup the layer. This creates and initializes the filters and biases, all filter weights with a value of 0.5, all biases with a value of 0.
		l1.initLayer();

		// Create the second layer.
		Layer l2 = new Layer(36, 36, 16, 5, 5, 16, 16, 1, 0, LayerType.CONV);
		l2.initLayer();

		// Create and initialize the third layer.
		Layer l3 = new Layer(32, 32, 16, 8, 8, 1, 256, 8, 0, LayerType.MAXPOOL);
		l3.initLayer();

		// Create and initialize the fourth layer.
		Layer l4 = new Layer(4, 4, 16, 2, 2, 16, 9, 1, 0, LayerType.LOCAL);
		l4.initLayer();
		
		// Create and initialize the fourth layer.
		Layer l5 = new Layer(3, 3, 1, 3, 3, 1, 4, 1, 0, LayerType.FULLY);
		l5.initLayer();
		

		// This is the last "layer": this will hold the output of the network
		fn.out = new Cell[4];

		// Initialize the cells because java won't do it for you
		for (int i = 0; i < 4; i++) {
			fn.out[i] = new Cell();
		}

		// Initialize the list of layers
		fn.layers = new LinkedList<Layer>();

		// Add each layer at the appropriate place in the list.
		fn.layers.add(0, l1);
		fn.layers.add(1, l2);
		fn.layers.add(2, l3);
		fn.layers.add(3, l4);
		fn.layers.add(4, l5);

		return fn;
	}

	public void learnFlag() throws Exception{
		
		// This array will hold the filenames of the pictures of each input file, 
		String[] learnInput = new String[80];
		
		for (int i=0; i<20; i++){
			String filename = "testingInput/flag/us"+(i+1)+".jpg";
			learnInput[i] = filename;
		}
		
		for (int i=0; i<20; i++){
			String filename = "testingInput/flag/uk"+(i+1)+".jpg";
			learnInput[i+20] = filename;
		}
		
		for (int i=0; i<20; i++){
			String filename = "testingInput/flag/irish"+(i+1)+".jpg";
			learnInput[i+40] = filename;
		}
		
		for (int i=0; i<20; i++){
			String filename = "testingInput/flag/greek"+(i+1)+".jpg";
			learnInput[i+60] = filename;
		}
		
		
		// This is the dictionary used to tell the learning functions what the ideal
		// output for a picture of that person would look like
		double[][] dictionary = new double[80][4];
		
		// Given an index, "dictionary[x][y]", 
		// x is the index in the list of names of the person the input represents
		// and y is the array of output we would expect to see in a perfectly trained network.
		// Since this is a simplified version, this networks "out" array is only 5 cells long,
		// so we can't expect it to recognize all the faces. 
		// So I've given it 4 faces to recognize and a 5th cell to indicate the input does
		// not represent any of those four faces.
		for(int k = 0; k <dictionary.length; k++){
			if (k <20){
				dictionary[k][0] = 1.0;
				dictionary[k][1] = 0.0;
				dictionary[k][2] = 0.0;
				dictionary[k][3] = 0.0;
			} else if ((k>=20)&& (k <40)){
				dictionary[k][0] = 0.0;
				dictionary[k][1] = 1.0;
				dictionary[k][2] = 0.0;
				dictionary[k][3] = 0.0;		
			}else if ((k>=40)&& (k <60)){
				dictionary[k][0] = 0.0;
				dictionary[k][1] = 0.0;
				dictionary[k][2] = 1.0;
				dictionary[k][3] = 0.0;		
			}else{
				dictionary[k][0] = 0.0;
				dictionary[k][1] = 0.0;
				dictionary[k][2] = 0.0;
				dictionary[k][3] = 1.0;		
			}
		}
		
		FSONNetwork.openFileInput(layers, learnInput[0]);
		
		double test[][][]= new double[layers.getFirst().depth][layers.getFirst().rows][layers.getFirst().collumns];
		// Depth
		for (int l = 0; l < layers.getFirst().depth; l++) {
			// Row
			for (int j = 0; j < layers.getFirst().rows; j++) {
				// Column
				for (int k = 0; k < layers.getFirst().collumns; k++) {
					test[l][j][k]= layers.getFirst().cells[l][j][k].value;
				}
			}
		}
		FSONNetwork.feedForward(layers, out, true);
		// Use the learning function to learn using our newly processed input and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		learn(1, this.layers, this.out, learnInput, 1, dictionary, false, "testlearnlfw.txt");
		
	}

	public static FSONNetwork patternNetwork() {
		
		FSONNetwork pn = new FSONNetwork();

		// Create the second layer.
		Layer l0 = new Layer(15, 15, 3, 5, 5, 3, 3, 1, 0, LayerType.CONV);
		l0.initLayer();

		// Create the second layer.
		Layer l1 = new Layer(11, 11, 3, 4, 4, 3, 3, 1, 0, LayerType.CONV);
		l1.initLayer();

		// Create the second layer.
		Layer l2 = new Layer(8, 8, 3, 3, 3, 3, 108, 1, 0, LayerType.MAXPOOL);
		l2.initLayer();

		// Create and initialize the third layer.
		Layer l3 = new Layer(6, 6, 3, 2 , 2, 3, 9, 2, 0, LayerType.LOCAL);
		l3.initLayer();

		// Create and initialize the fourth layer.
		Layer l4 = new Layer(3, 3, 1, 3, 3, 1, 1, 1, 0, LayerType.FULLY);
		l4.initLayer();
		

		// This is the last "layer": this will hold the output of the network
		pn.out = new Cell[1];

		// Initialize the cells because java won't do it for you
		pn.out[0] = new Cell();

		// Initialize the list of layers
		pn.layers = new LinkedList<Layer>();

		// Add each layer at the appropriate place in the list.
		pn.layers.add(0, l0);
		pn.layers.add(1, l1);
		pn.layers.add(2, l2);
		pn.layers.add(3, l3);
		pn.layers.add(4, l4);
		
		return pn;
	}

	public void learnPattern() throws Exception{
		
		// This array will hold the filenames of the pictures of each input file, 
		String[] learnInput = new String[14];
		
		for (int i=0; i<14; i++){
			String filename = "testingInput/patterns/easy/"+i+".jpg";
			learnInput[i] = filename;
		}
		
		
		// This is the dictionary used to tell the learning functions what the ideal
		// output for a picture of that person would look like
		double[][] dictionary = new double[14][1];
		
		// Given an index, "dictionary[x][y]", 
		// x is the index in the list of names of the person the input represents
		// and y is the array of output we would expect to see in a perfectly trained network.
		dictionary[0][0] = 0.0;
		dictionary[1][0] = 0.0;
		dictionary[2][0] = 0.0;
		dictionary[3][0] = 0.0;
		dictionary[4][0] = 0.0;
		dictionary[5][0] = 0.0;
		dictionary[6][0] = 1.0;
		dictionary[7][0] = 0.0;
		dictionary[8][0] = 0.0;
		dictionary[9][0] = 1.0;
		dictionary[10][0] = 1.0;
		dictionary[11][0] = 0.0;
		dictionary[12][0] = 0.0;
		dictionary[13][0] = 1.0;
		
		FSONNetwork.openFileInput(layers, learnInput[0]);
		
//		double test[][][]= new double[layers.getFirst().depth][layers.getFirst().rows][layers.getFirst().collumns];
//		// Depth
//		for (int l = 0; l < layers.getFirst().depth; l++) {
//			// Row
//			for (int j = 0; j < layers.getFirst().rows; j++) {
//				// Column
//				for (int k = 0; k < layers.getFirst().collumns; k++) {
//					test[l][j][k]= layers.getFirst().cells[l][j][k].value;
//				}
//			}
//		}
		
		FSONNetwork.feedForward(layers, out, true);
		
		// Use the learning function to learn using our newly processed input and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		learn(1, this.layers, this.out, learnInput, 1000, dictionary, true, "testlearnpattern.txt");
		
	}
}