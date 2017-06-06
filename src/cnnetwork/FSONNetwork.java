package cnnetwork;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
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
	 * Please note that since the only time we are looking for a parial derivative 
	 * with respect to a weight for backpropagation is when we are looking to 
	 * increment that paticular weight, the layer in which that weight resides
	 * cannot be a maxpool layer.
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
	 *             This exception may also be thrown when trying to use this
	 *             function to find the partial derivative for a weight that
	 *             resides in a maxpool layer.
	 *             
	 */
	public static double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, int filterIndex,
			int depth, int row, int column, double[] expected) throws Exception {
		// If there is already a value stored for this partial derivative...
		if (!Double.isNaN(layers.get(layerIndex).filters.get(filterIndex).gradientValues[depth][row][column])) {
			// ...then just use that
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
				Double dnetdw = layers.get(layerIndex).cells[startCell.depth + depth][startCell.row
						+ rowTrue][startCell.column + columnTrue].value;

				// If this is the last layer (IE the layer before "out"):
				if ((layerIndex + 1) == layers.size()) {
					sum += (dnetdw * computeSoftmaxError(out, filterIndex, expected));
				} else {

					// Since the only time we are looking for a parial derivative with respect to
					// a *weight* is when we are looking to increment that paticular weight,
					// the layer in which that weight resides *cannot* be a maxpool layer.
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
	 * Please note that since biases are not used in maxpool layers, the layer
	 * in which the bias resides cannot be a maxpool layer.
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
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias that resides in a maxpool
	 *             layer. Since biases are not used in maxpool layers, the layer
	 * 			   in which the bias resides cannot be a maxpool layer.
	 * 
	 */
	public static double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, int biasIndex,
			double[] expected) throws Exception {
		// If there is already a value stored for this partial derivative...
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
							
							// Since biases are not used in maxpool layers, the layer 
							// in which the bias resides cannot be a maxpool layer.
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

					// If this filter is applied to the cell we are looking at...
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
	 *             "computePartialDerivative" function(s).
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
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
	 *             "computePartialDerivative" function(s).
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
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
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input. See openFileInput (declared in this class file) for
	 *             more details.
	 * 
	 * TODO: Add functionality to enable/disable console output.
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
		System.out.println("0," + totalError);
		
		// Calculate the starting learning rate using the starting error and the learningFactor parameter
		double learningRate;
		
		if (totalError == Double.NEGATIVE_INFINITY){
			learningRate = learningFactor;
		} else{
			learningRate = Layer.activationFunction(totalError) * learningFactor;
		}

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
							// it.
							// TODO: Add option to change this to RGB
							openHSVFileInput(layers, inputs[n]);
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
//							System.out.println("----------------------------------------------");
//							System.out.println("Processing layer: " + j);
//							System.out.println("----------------------------------------------");
							Layer currentLayer = layers.get(j);
							
							if (currentLayer.type!=LayerType.MAXPOOL){
								// 2.a.i) Increment all weights for all the filters for this layer ("currentLayer")
								for (int f = 0; f < currentLayer.filters.size(); f++) {
//									System.out.println("Incrementing filter: " + f);
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
//									System.out.println("Incrementing bias: "+ b);
									currentLayer.biases.get(b).previousValue = currentLayer.biases.get(b).value;
									currentLayer.biases.get(b).value = stepGradient(learningRate, layers, out, j, b, dictionary[s]);
								}
							}

						}

						
//						System.out.println("----------------------------------------------");
//						System.out.println("Recording and resetting layers");
//						System.out.println("----------------------------------------------");
						
						// Open a file, using the path created above, to store our progress while learning.
						// This file can then be used to recover a network if we are interrupted while learning.
						PrintWriter fw = new PrintWriter(absPath);
						
						// 3. Reset stored gradients and write the new weights to the file
						// 3.a) Reset all stored gradients for all layers
						for (int j = 0; j < layers.size(); j++) {
//							System.out.println("----------------------------------------------");
//							System.out.println("Recording layer: "+ j);
//							System.out.println("----------------------------------------------");
							// Record this layer
							fw.write("<layer>\n");
							
							Layer currentLayer = layers.get(j);

							// Record the paramaters for this layer
							fw.write(currentLayer.collumns + "," + currentLayer.rows+ "," + currentLayer.depth + "," + currentLayer.Fcollumns + "," + currentLayer.Frows + "," + currentLayer.Fdepth + "," + currentLayer.K + ","+ currentLayer.step+ ","+ currentLayer.pad+ "," + currentLayer.type+"\n");
							fw.flush();

							// Record the cells of this layer
							fw.write("<cells>\n");
							
//							System.out.println("Recording cells");
							
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
//									System.out.println("Recording filter: "+f );
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
//									System.out.println("Recording bias: "+b );
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
					if (totalError == Double.NEGATIVE_INFINITY){
						learningRate = learningFactor;
					} else{
						learningRate = Layer.activationFunction(totalError) * learningFactor;
					}
					
				}
				
			}
			
//			System.out.println("----------------------------------------------");
//			System.out.println("Recalculating error");
//			System.out.println("----------------------------------------------");
		
//			if(i%5 == 0){
//				System.out.println((i+1) + "," + totalError);
//			}
				
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
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input.
	 * 
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
					// TODO: Add functionality to control this formatting, since this can be used to tune input to benefit learning.
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
	 * network. Input files are expected to be in color and in .jpg form.
	 * Color format is HSV, where all values range from 0 to 1 (inclusive).
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
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input.
	 *             
	 */
	public static void openHSVFileInput(LinkedList<Layer> layers, String filename) throws Exception {
		
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
		
		//Convert the Mat into HSV form
		Mat hsvMat = new Mat();
		Imgproc.cvtColor(resizedImage, hsvMat, Imgproc.COLOR_BGR2HSV);

		// Split the image into desired number of color channels
		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here in the order RGB
		Core.split(hsvMat, channels);

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
					
					// In OpenCV's  HSV, Hue range is [0,179], Saturation range is 
					// [0,255] and Value range is [0,255].
					// Since the network is expecting a value between 0 and 1, some
					// formatting of the data is required.
					// TODO: Add functionality to control this formatting, since this can be used to tune input to benefit learning.
					double value = temp[(d * layers.get(0).rows) + e];
					if (c == 0){ //If we are processing the Hue channel...
						value /= 179.0;
					} else {
						// For now, discard the Saturation and Value information.
						// TODO: Add functionality to incorperate Saturation and Value into learning.
						value = 0;
					}
					
					layers.get(0).cells[c][d][e].value = value;
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
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
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
				// TODO: Add functionality to control this formatting, since this can be used to tune input to benefit learning.
				double value =resizedImage.get(d,e)[0];
				value = value/25.5;
				layers.get(0).cells[0][d][e].value = value;
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
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
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
	 *             This exception is also thrown when trying to find the partial
	 *             derivative with respect to a bias or weight that resides in 
	 *             a maxpool layer. Since biases are not used in maxpool layers,
	 *             and weights cannot be updated in a maxpool layer, the layer
	 * 			   in which the bias or weight resides cannot be a maxpool layer.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input.
	 *             
	 */
	public static double crossEntropyTotalError(LinkedList<Layer> layers, Cell[] out, String[] input,
			double[][] dictionary, boolean independent) throws Exception {
		
		double sum = 0;
		int count = 0;

		// For each example in the training data...
		for (int i = 0; i < input.length; i++) {

			//If there is an input to be learned at this index...
			if ((input[i] != null) && (!input[i].equals(""))){
				
				String[] inputs = input[i].split(",");
				int numInputs = inputs.length;
				for (int n = 0; n<numInputs; n++){
					count++;
					// Open that example and feed it through the network:
					// If the first layer only has a depth of 1, that means the input is
					// supposed to be black and white, so use the appropriate function
					// to open it
					if (layers.getFirst().cells.length == 1) {
						openFileInputBW(layers, inputs[n]);
					} else { // If the first layer has more than a single depth, that
								// means it is expecting an image with multiple
								// channels, so use the appropriate function to open it
								// TODO: Add option to change this to RGB.
						openHSVFileInput(layers, inputs[n]);
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
						// TODO: Document this section better.
						if (out[k].value != dictionary[i][k]) {
							double log;
							if ((out[k].value == 0) && (dictionary[i][k] != 0)) {
								//TODO: fix this
								sum = Double.NEGATIVE_INFINITY;
								return sum;
							} else {
								log = Math.log(out[k].value);
							}

							double oneMinusLog;

							if ((out[k].value == 1) && (dictionary[i][k] != 1)) {
								//TODO: fix this
								sum = Double.NEGATIVE_INFINITY;
								return sum;

							} else {
								oneMinusLog = Math.log(1 - out[k].value);
							}

							sum += (dictionary[i][k] * log) + ((1 - dictionary[i][k]) * oneMinusLog);
						}

					}
				}
			}

		}
		// Multiply sum by -1/n, where n is the number of examples in the training data
		double error = (sum * (0.0 - (1.0)))/(double) count;

		return error;
	}
	
	//TODO: Document or take this out.
	public static double crossEntropyTotalErrorArray(double[] actual, double[] expected){
		double sum = 0.0;
		for (int k = 0; k < actual.length; k++) {
			// Add y*ln(x) + (1-y)*ln(1-x) to the sum,
			// where y is the expected value for this cell
			// and x is the actual value for this cell
			if (actual[k] != expected[k]) {
				double log;
				if ((actual[k] == 0) && (expected[k] != 0)) {
					//TODO: Fix this.
					sum = Double.NEGATIVE_INFINITY;
					return sum;
				} else {
					log = Math.log(actual[k]);
				}
				double oneMinusLog;

				if ((actual[k] == 1) && (expected[k] != 1)) {
					sum = Double.NEGATIVE_INFINITY;
					return sum;

				} else {
					oneMinusLog = Math.log(1 - actual[k]);
				}

				sum += (expected[k] * log) + ((1 - expected[k]) * oneMinusLog);
			}
		}
		
		// Multiply sum by -1/n, where n is the number of examples in the training data
		double error = (sum * (0.0 - (1.0)))/(double) actual.length;

		return error;
		
	}
	
	/**
	 * This function creates, initializes, and returns a sample network
	 * designed to recognize color.
	 * 
	 * The network has an output array of 9 doubles, each representing the
	 * probability that the input color corresponds to the 8 colors:
	 * 
	 * Red, Orange, Yellow, Green, Aqua (blue/green), Blue, Purple,
	 * and Pink (fuscha). 
	 * 
	 * (in that order)
	 * 
	 * The last element in the array also corresponds to red; This is an 
	 * artifact of the arrangement of the Hue, where both 0 and 360 
	 * correspond to the same canonical red.
	 * 
	 * Please note that the saturation and value (IE "brightness") are
	 * disregarded (this means that the colors "white", "black", and 
	 * "grey" are not recognized).
	 * 
	 * Please also note that the input layer takes only a single cell 
	 * (of depth 3, corresponding to Hue, Saturation, and Value, in that
	 * order) as input, so any colors detected will be the average of
	 * all colors in the input image.
	 * 
	 * @return A sample color-recognition network, initialized but untrained.
	 */
	public static FSONNetwork colorNetwork() {
		
		FSONNetwork cn = new FSONNetwork();

		Layer l0 = new Layer(1, 1, 3, 1, 1, 3, 9, 1, 0, LayerType.CONV);
		l0.initLayer();
		
		Layer l1 = new Layer(1, 1, 9, 1, 1, 9, 9, 1, 0, LayerType.FULLY);
		l1.initLayer();
		
		// This is the last "layer": this will hold the output of the network
		cn.out = new Cell[9];

		// Initialize the cells because java won't do it for you
		cn.out[0] = new Cell();
		cn.out[1] = new Cell();
		cn.out[2] = new Cell();
		cn.out[3] = new Cell();
		cn.out[4] = new Cell();
		cn.out[5] = new Cell();
		cn.out[6] = new Cell();
		cn.out[7] = new Cell();
		cn.out[8] = new Cell();
		
		// Initialize the list of layers
		cn.layers = new LinkedList<Layer>();

		// Add each layer at the appropriate place in the list.
		cn.layers.add(0, l0);
		cn.layers.add(1, l1);
		
		return cn;
	}

	/**
	 * This function creates and trains a sample network to detect color. The
	 * network has an output array of 9 doubles, each representing the
	 * probability that the input color corresponds to the 8 colors:
	 * 
	 * Red, Orange, Yellow, Green, Aqua (blue/green), Blue, Purple, and Pink
	 * (fuscha).
	 * 
	 * (in that order)
	 * 
	 * The last element in the array also corresponds to red; This is an
	 * artifact of the arrangement of the Hue, where both 0 and 360 correspond
	 * to the same canonical red.
	 * 
	 * Please note that the saturation and value (IE "brightness") are
	 * disregarded (this means that the colors "white", "black", and "grey" are
	 * not recognized).
	 * 
	 * See the function: public static FSONNetwork colorNetwork() ...declared in
	 * this class for more details on the network itself.
	 * 
	 * @return A sample trained color-recognition network.
	 * @throws Exception
	 *             Thrown when there is a problem reading the training input
	 *             files.
	 */
	public static FSONNetwork createAndTrainColorNetwork() throws Exception{
		FSONNetwork colorNet = FSONNetwork.colorNetwork();

		// This array will hold the filenames of the pictures of each input file
		String[] learnInput = new String[18];
		
		learnInput[0] = "testingInput/colors/red/0.jpg";
		learnInput[1] = "testingInput/colors/orange/0.jpg";
		learnInput[2] = "testingInput/colors/yellow/0.jpg";
		learnInput[3] = "testingInput/colors/green/0.jpg";
		learnInput[4] = "testingInput/colors/aqua/0.jpg";
		learnInput[5] = "testingInput/colors/blue/0.jpg";
		learnInput[6] = "testingInput/colors/purple/0.jpg";
		learnInput[7] = "testingInput/colors/pink/0.jpg";
		learnInput[8] = "testingInput/colors/red/1.jpg";
		learnInput[9] = "testingInput/colors/red/2.jpg";
		learnInput[10] = "testingInput/colors/orange/1.jpg";
		learnInput[11] = "testingInput/colors/yellow/1.jpg";
		learnInput[12] = "testingInput/colors/green/1.jpg";
		learnInput[13] = "testingInput/colors/aqua/1.jpg";
		learnInput[14] = "testingInput/colors/blue/1.jpg";
		learnInput[15] = "testingInput/colors/purple/1.jpg";
		learnInput[16] = "testingInput/colors/pink/1.jpg";
		learnInput[17] = "testingInput/colors/red/3.jpg";
		
		
		// This is the dictionary used to tell the learning functions what the
		// ideal output for a picture of that person would look like
		double[][] dictionary = new double[18][9];

		// Given an index, "dictionary[x][y]",
		// x is the index in the list of names of the person the input
		// represents, 
		// and y is the array of output we would expect to see in a perfectly
		// trained network.
		dictionary[0][0] =1.0;
		dictionary[1][1] =1.0;
		dictionary[2][2] =1.0;
		dictionary[3][3] =1.0;
		dictionary[4][4] =1.0;
		dictionary[5][5] =1.0;
		dictionary[6][6] =1.0;
		dictionary[7][7] =1.0;
		dictionary[8][8] =1.0;
		dictionary[9][0] =1.0;
		dictionary[10][1] =1.0;
		dictionary[11][2] =1.0;
		dictionary[12][3] =1.0;
		dictionary[13][4] =1.0;
		dictionary[14][5] =1.0;
		dictionary[15][6] =1.0;
		dictionary[16][7] =1.0;
		dictionary[17][8] =1.0;
		
		// Open and feed forward the first training input to create the connections
		// used in backpropagation (training).
		FSONNetwork.openHSVFileInput(colorNet.layers, learnInput[0]);
		FSONNetwork.feedForward(colorNet.layers, colorNet.out, true);

		// Use the learning function to learn using our newly processed input
		// and newly created dictionary.
		// Use the file "testlearnlfw.txt" to store our progress while learning.
		// The learning factor is set low here because input ranges from 0 to 1 
		// over a single cell, so finer control is needed.
		FSONNetwork.learn(0.225, colorNet.layers, colorNet.out, learnInput, 15000, dictionary, false,
				"testlearncolor.txt");
		
		return colorNet;
	}
	
	/**
	 * This function creates, initializes, and returns a sample network
	 * designed to recognize the presence of a simple circle  in a 
	 * greyscale input image.
	 * 
	 * @return A sample shape-recognition network, initialized but untrained.
	 */
	public static FSONNetwork shapeNetwork() {
		
		FSONNetwork sn = new FSONNetwork();

		Layer l0 = new Layer(20, 20, 1, 7, 7, 1, 1, 1, 0, LayerType.CONV);
		l0.initLayer();
		
		Layer l2 = new Layer(14, 14, 1, 14, 14, 1, 1, 1, 0, LayerType.FULLY);
		l2.initLayer();
		
		// This is the last "layer": this will hold the output of the network
		sn.out = new Cell[1];

		// Initialize the cells because java won't do it for you
		sn.out[0] = new Cell();
		
		// Initialize the list of layers
		sn.layers = new LinkedList<Layer>();

		// Add each layer at the appropriate place in the list.
		sn.layers.add(0, l0);
		sn.layers.add(1, l2);
		
		return sn;
	}
}