package cnnetwork;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import image.Align;

/**
 * This class is a simple demonstration of the FSON system.
 * 
 * @author Paula Rudy
 *
 */
public class FSONNetwork {

	public LinkedList<Layer> layers; //The layers that make up this sample network
	public Cell[] out; //This is the last layer of this network, the "output".
	
	/**
	 * This function creates and sets up the network. 
	 * There are 8 layers and an output layer.
	 * Layers 1,3, and 5 are all convoltional layers.
	 * Layers 2 and 4 are maxpool layers.
	 * Layers 6 and 7 are locally connected layers,
	 * and layer 8 is a fully connected layer.
	 * The weights of all the filters are either 1 or 0.5 depending
	 * on what layer they are for.
	 * 
	 */
	public FSONNetwork() {

		//Declare and initialize the first layer
		Layer l1 = new Layer(76, 76, 3, 5, 5, 3, 32, 1, 0, LayerType.CONV);

		//Setup the layer. This creates and initializes the filters and biases, all with a value of 1.
		l1.initLayer();

		//Create the second layer.
		Layer l2 = new Layer(72, 72, 32, 3, 3, 32, 39200, 2, 0, LayerType.MAXPOOL);
		l2.initLayer();
		
		//Create and initialize the third layer.
		Layer l3 = new Layer(35, 35, 32, 5, 5, 32, 16, 1, 0, LayerType.CONV);
		l3.initLayer();

		//Create the fourth layer.
		Layer l4 = new Layer(31, 31, 16, 5, 5, 16, 3136, 2, 0, LayerType.MAXPOOL);
		l4.initLayer();
		
		//Create and initialize the fifth layer.
		Layer l5 = new Layer(14, 14, 16, 3, 3, 16, 16, 2, 0, LayerType.CONV);
		l5.initLayer();

		//Create and initialize the sixth layer.
		//This one is a locally connected layer.
		Layer l6 = new Layer(6, 6, 16, 3, 3, 1, 400, 1, 0, LayerType.LOCAL);
		l6.initLayer();

		//Create and initialize the seventh layer.
		//This one is also a locally connected layer.
		Layer l7 = new Layer(5, 5, 16, 5, 5, 16, 2048, 1, 0, LayerType.FULLY);
		l7.initLayer();

		//Create and initialize the eighth layer.
		//This one is a fully connected layer.
		Layer l8 = new Layer(2048, 1, 1, 2048, 1, 1, 2016, 1, 0, LayerType.FULLY);
		l8.initLayer();

		//This is the last "layer": this will hold the output of the network
		this.out = new Cell[2016];
		
		//Initialize the cells because java won't do it for you
		for (int i = 0; i < 2016; i++) {
			this.out[i] = new Cell();
		}

		//Initialize the list of layers
		this.layers = new LinkedList<Layer>();

		//Add each layer at the appropriate place in the list.
		this.layers.add(0, l1);
		this.layers.add(1, l2);
		this.layers.add(2, l3);
		this.layers.add(3, l4);
		this.layers.add(4, l5);
		this.layers.add(5, l6);
		this.layers.add(6, l7);
		this.layers.add(7, l8);

	}

	/**
	 * This function feeds a buffered image through the network,
	 * 
	 * @param image
	 *            The image to use as input
	 * @throws Exception
	 *             Thrown when the activation function does not return a number
	 *             (see Layer::activationFunction()).
	 */
	public void calculate(BufferedImage image) throws Exception {

		//This is necessary to use any of the OpenCV functions
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		//First, convert the image to an OpenCV Mat.
		//This function can be found in image.Align.java
		Mat test = Align.bufferedImageToMat(image);

		//Next, resize the image to the size needed.
		Mat resizedImage = new Mat();
		Size sz = new Size(76, 76);
		Imgproc.resize(test, resizedImage, sz);

		//Split the image into three color channels
		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here
													// in the order RGB
		Core.split(resizedImage, channels);

		//For each channel...
		for (int i = 0; i < channels.size(); i++) {

			//Feed the individual pixel values into a temporary array...
			channels.get(i).convertTo(channels.get(i), CvType.CV_64FC3);
			int size = (int) (channels.get(i).total() * channels.get(i).channels());
			double[] temp = new double[size];
			channels.get(i).get(0, 0, temp);
			
			//...and then into the cells of the first layer of the network.
			for (int j = 0; j < 76; j++) {
				for (int k = 0; k < 76; k++) {
					this.layers.get(0).cells[i][j][k].value = temp[(j * 76) + k];
				}

			}

		}

		//Call the appropriate functions to feed the input through the layers
		this.layers.get(0).convolution(this.layers.get(0).cells, this.layers.get(0).filters, this.layers.get(1).cells, this.layers.get(0).step, this.layers.get(0).pad, this.layers.get(0).biases, true);
		this.layers.get(1).pool(this.layers.get(1).cells, this.layers.get(1).filters, this.layers.get(2).cells, this.layers.get(1).step, this.layers.get(1).Fcollumns, true);
		this.layers.get(2).convolution(this.layers.get(2).cells, this.layers.get(2).filters, this.layers.get(3).cells, this.layers.get(2).step, this.layers.get(2).pad, this.layers.get(2).biases, true);
		this.layers.get(3).pool(this.layers.get(3).cells, this.layers.get(3).filters, this.layers.get(4).cells, this.layers.get(3).step, this.layers.get(3).Fcollumns, true);
		this.layers.get(4).convolution(this.layers.get(4).cells, this.layers.get(4).filters, this.layers.get(5).cells, this.layers.get(4).step, this.layers.get(4).pad, this.layers.get(4).biases, true);
		this.layers.get(5).local(this.layers.get(5).cells, this.layers.get(5).filters, this.layers.get(6).cells, this.layers.get(5).step, this.layers.get(5).pad, this.layers.get(5).biases, true);
		this.layers.get(6).full(this.layers.get(6).cells, this.layers.get(6).filters, this.layers.get(7).cells[0][0], this.layers.get(6).step, this.layers.get(6).pad, this.layers.get(6).biases, true);
		this.layers.get(7).full(this.layers.get(7).cells, this.layers.get(7).filters, this.out, this.layers.get(7).step, this.layers.get(7).pad, this.layers.get(7).biases, true);
		Layer.softmax(this.out);
	}

	/**
	 * This function computes the partial derivative of the total error with
	 * respect to a given *weight* within a network. This is a recursive function
	 * that operates with the help of the other function with this name and the
	 * call signature:
	 * 
	 * computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int
	 * layerIndex, CellCoord outcell, double[] expected)
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
	 *            The array of cells that represent the expected values of "out".
	 * @return The calculated partial derivative of the total error with respect
	 *         to a given weight.
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell, via the
	 *             compute function used on the line marked
	 *             "//Find the net value of the output cell". See
	 *             Layer::activationFunction() for more details.
	 */
	public static double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, int filterIndex, int depth, int row, int column, double[] expected) throws Exception{
		// If there is already a value stored for this partial derivative...
		if (layers.get(layerIndex).filters.get(filterIndex).gradientValues[depth][row][column] != -1) {
			// ...then just use that
			return layers.get(layerIndex).filters.get(filterIndex).gradientValues[depth][row][column];
		} else {

			// This will hold the sum of all the calculated partial derivatives
			// of all the connections that use this filter
			// (with respect to this weight)
			// IE: Sum( dNet(i) /dw) 
			// where i = the net value of each connection's outcell
			double sum = 0;

			// For each connection for this filter (and thus uses this weight in
			// it's calculations)...
			for (int i = 0; i < layers.get(layerIndex).filters.get(filterIndex).connections.size(); i++) {
				// Calculate dnet/dw:

				// First, grab the FilterConnection associated with this filter
				// that we are dealing with right now.
				FilterConnection thisConnection = layers.get(layerIndex).filters.get(filterIndex).connections.get(i);

				// Find the value of the cell associated with this weight:
				// This is the CellCoordinate of the first cell in this layer
				// used in this calculation
				CellCoord startCell = thisConnection.inStart;

				// This is to compensate for odd shaped filters and/or layers;
				// if the filter and/or layer lacks a dimension, they will be
				// given as -1,
				// so we will count those as 0 to avoid altering where we are
				// going to look for the cell.
				// TODO: Take this out? 
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
				// (the cell's value multiplied by this weight when calculating
				// the net value for this connection).
				// Since this is the only thing multiplied by the weight when
				// calculating, that means that this value is *also* the
				// derivative of
				// the net of this output cell with respect to this weight(IE
				// dnet/dw).
				// TODO: Store this instead of calculate it?
				Double dnetdw = layers.get(layerIndex).cells[startCell.depth + depth][startCell.row
						+ rowTrue][startCell.column + columnTrue].value;

				if (((layerIndex + 1) < layers.size()) && (layers.get(layerIndex
						+ 1).cells[thisConnection.out.depth][thisConnection.out.row][thisConnection.out.column].derivative != -1)) {
					// If this is NOT the last layer, but there is a derivative
					// value already stored in the "out" cell for this
					// connection, use that.
					sum += (dnetdw * layers.get(layerIndex
							+ 1).cells[thisConnection.out.depth][thisConnection.out.row][thisConnection.out.column].derivative);
				} else {

					// Find the net value of the output cell:
					double cellOut = Layer.compute(layers.get(layerIndex).filters.get(filterIndex),
							layers.get(layerIndex).cells, startCell.column, startCell.row, startCell.depth,
							layers.get(layerIndex).biases.get(filterIndex));

					// This is the calculated derivative of the out value with
					// respect to net for this cell
					// IE dout/dnet
					double doutdnet = cellOut * (1 - cellOut);

					// If this is the last layer (IE the layer before "out"):
					if ((layerIndex + 1) == layers.size()) {
						sum += (dnetdw * doutdnet * computeSoftmaxError(out, thisConnection.out.depth, expected));
					} else {
						// Continue to recursively calculate the derivative
						sum += (dnetdw * doutdnet * computePartialDerivative(layers, out, (layerIndex + 1),
								thisConnection.out, expected));
					}

				}

			}

			return sum;
		}

	}
	
	/**
	 * This function computes the partial derivative of the total error with
	 * respect to a given *bias* within a network. This is a recursive function
	 * that operates with the help of the other function with this name and the
	 * call signature:
	 * 
	 * computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int
	 * layerIndex, CellCoord outcell, double[] expected)
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
	 * TODO: error checking to make sure we're not on a maxpool layer
	 */
	public static double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex, int biasIndex, double[] expected) throws Exception{
		// If there is already a value stored for this partial derivative...
		if (layers.get(layerIndex).biases.get(biasIndex).derivative != -1) {
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

				//For each connection for this filter
				for (int j = 0; j < currentFilter.connections.size(); j++){
					FilterConnection currentConnection = currentFilter.connections.get(j);

					//If this connection uses the bias we are looking at...
					if (currentConnection.biasIndex == biasIndex){

						// Since 1 is the only thing multiplied by the bias when
						// calculating, that means that 1 is the derivative of
						// the net of this output cell with respect to this bias(IE
						// dnet/dbias). 
						// 
						// So we can ignore dnet/dbias in our calculations.

						// If this is NOT the last layer, but there is a derivative
						// value already stored in the "out" cell for this
						// connection...
						if (((layerIndex + 1) < layers.size()) && (layers.get(layerIndex
								+ 1).cells[currentConnection.out.depth][currentConnection.out.row][currentConnection.out.column].derivative != -1)) {
							
							// ... use that stored derivative value.
							//
							// Since 1 is the only thing multiplied by the bias when
							// calculating, that means that 1 is the derivative of
							// the net of this output cell with respect to this bias(IE
							// dnet/dbias). So we can ignore dnet/dbias in our 
							// calculations.
							sum += (layers.get(layerIndex
									+ 1).cells[currentConnection.out.depth][currentConnection.out.row][currentConnection.out.column].derivative);
						} else {

							// Find the net value of the output cell:
							double cellOut = Layer.compute(layers.get(layerIndex).filters.get(biasIndex),
									layers.get(layerIndex).cells, currentConnection.inStart.column,
									currentConnection.inStart.row, currentConnection.inStart.depth,
									layers.get(layerIndex).biases.get(biasIndex));

							// This is the calculated derivative of the out value with
							// respect to net for this cell. IE dout/dnet
							double doutdnet = cellOut * (1 - cellOut);

							// If this is the last layer (IE the layer before "out"):
							if ((layerIndex + 1) == layers.size()) {
								
								// Note that since 1 is the only thing
								// multiplied by the bias when calculating, that
								// means that 1 is the derivative of the net of
								// this output cell with respect to this bias(IE
								// dnet/dbias). So we can ignore dnet/dbias in
								// our calculations.
								sum += (doutdnet * computeSoftmaxError(out, currentConnection.out.depth, expected));
								
							} else {
								
								// Continue to recursively calculate the derivative
								//
								// Note that since 1 is the only thing
								// multiplied by the bias when calculating, that
								// means that 1 is the derivative of the net of
								// this output cell with respect to this bias(IE
								// dnet/dbias). So we can ignore dnet/dbias in
								// our calculations.
								sum += (doutdnet * computePartialDerivative(layers, out, (layerIndex + 1),
										currentConnection.out, expected));
							}
						}

					}
				}
			}
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
	 *             ...for more details. TODO: store derivatives
	 */
	private static Double computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int layerIndex,
			CellCoord outcell, double[] expected) throws Exception {

		// This will hold the sum of all relevant partial derivatives;
		// IE sum(dtotalerror/dnet_i)
		// where i is all the weights that are applied to this cell,
		// and total error is the total error of the whole network 
		// with respect to the given expected values in double[]
		// expected.
		double dEdnet = 0;

		// For all FilterConnections that use this cell for input...
		for (int i = 0; i < layers.get(layerIndex).K; i++) {
			LinkedList<FilterConnection> relevantConnections = layers.get(layerIndex).filters.get(i).connections;
			for (int j = 0; j < relevantConnections.size(); j++) {
				// TODO: Account for -1s?
				// If this filter is applied to the cell we are looking at...
				if ((relevantConnections.get(j).inStart.depth <= outcell.depth)
						&& (relevantConnections.get(j).inStart.row <= outcell.row)
						&& (relevantConnections.get(j).inStart.column <= outcell.column)
						&& ((relevantConnections.get(j).inStart.depth + layers.get(layerIndex).Fdepth) >= outcell.depth)
						&& ((relevantConnections.get(j).inStart.row + layers.get(layerIndex).Frows) >= outcell.row)
						&& ((relevantConnections.get(j).inStart.column
								+ layers.get(layerIndex).Fcollumns) >= outcell.column)) {

					// Grab the depth, row, and column in the filter of the
					// weight multiplied by this cell
					// when this connection is calculated
					int depth = (outcell.depth - relevantConnections.get(j).inStart.depth);
					int row = (outcell.row - relevantConnections.get(j).inStart.row);
					int column = (outcell.column - relevantConnections.get(j).inStart.column);

					// Find the derivative of the total error with respect to
					// that weight
					// (using computePartialDerivative(LinkedList<Layer> layers,
					// Cell[] out, int layerIndex, int filterIndex, int depth,
					// int row, int column, double[] expected),
					// and add the result to our sum variable.
					dEdnet += computePartialDerivative(layers, out, layerIndex, i, depth, row, column, expected);
				}
			}
		}

		return dEdnet;
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
	public static double stepGradient(double learningRate, LinkedList<Layer> layers, Cell[] out, int layerIndex, int filterIndex, int depth, int row, int column, double[] expected) throws Exception{
		double weight = layers.get(layerIndex).filters.get(filterIndex).weights[depth][row][column];
		double dEdweight = computePartialDerivative(layers, out, layerIndex, filterIndex, depth, row, column, expected);
		return (weight - (learningRate * dEdweight));
	}
	
	/**
	 * This function is used to increment a single bias during the learning process.
	 * 
	 * @param learningRate
	 *            The learning rate of the learning process. How "far" each
	 *            bias moves per increment.
	 * @param layers
	 *            The layers that make up this network
	 * @param out
	 *            The array of cells that store the output of this network.
	 * @param layerIndex
	 *            The index within "layers" of the layer in which the bias of
	 *            interest resides.
	 * @param biasIndex
	 *            The index within "biases" of the layer in which the bias resides.
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
	public static double stepGradient(double learningRate, LinkedList<Layer> layers, Cell[] out, int layerIndex, int biasIndex, double[] expected) throws Exception{
		double bias = layers.get(layerIndex).biases.get(biasIndex).value;
		double dEdbias = computePartialDerivative(layers, out, layerIndex, biasIndex, expected);
		return (bias - (learningRate * dEdbias));
	}
	
	/**
	 * This function is the main function from which learning occurs. It is a
	 * stochastic gradient descent model.
	 * 
	 * @param learningRate
	 *            The learning rate of the learning process. How "far" each bias
	 *            or weight moves per increment.
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
	 * @throws Exception
	 *             This exception is thrown when a problem occurs while
	 *             calculating the activation function for a cell. See
	 *             Layer::activationFunction() for more details.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file for
	 *             input. See openFileInput (declared in this class file) for
	 *             more details.
	 */
	public static void learn(double learningRate, LinkedList<Layer> layers, Cell[] out, String[] input, int iterations) throws Exception {

		//Generate the "dictionary". This is an array of the expected outputs for each input. Since the "perfect" expected output for a given input is simply an array with all the entries are 0 but the one associated with the correct classification (which is 1), we can fill this out beforehand.
		double[][] dictionary = new double[input.length][input.length];
		for (int t = 0; t < input.length; t++){
			dictionary[t][t] = 1;
		}
			
		//For the requested number of iterations...
		for (int i = 0; i < iterations; i++) {
			
			//1. Pick an example, feed it forward through the network:
			//1.a) Generate a random order in which to access the input:
			ArrayList<Integer> randomList = UniqueRandomNumbers.getRandomSet(input.length);
			
			//1.b) Use "randomList" to access each input entry in a random order
			for (int r=0; r < input.length; r++) {
				int s = randomList.get(r);
				
				//1.b.i) Open the next input file denoted by the string stored in the input array, in a random order, and feed that input into the first layer
				openFileInputBW(layers, input[s]);
				
				//1.b.ii) Feed the input through the rest of the network
				feedForward(layers, out, false);
				Layer.softmax(out);
				
				//2. Increment all weights:
				
				//2.a) Increment all weights for all the layers, working backward.
				for(int j = (layers.size() -1); j >= 0 ; j--){
					Layer currentLayer = layers.get(j);
					
					//2.a.i) Increment all weights for all the filters for this layer ("currentLayer")
					for(int f = 0; f< currentLayer.filters.size(); f++){
						Filter currentFilter = currentLayer.filters.get(f);
						
						//2.a.i.1) Increment each weight within this filter ("currentFilter")
						for (int x = 0; x < currentLayer.Fdepth; x++) {
							for (int y = 0; y < currentLayer.Frows; y++) {
								for (int z = 0; z < currentLayer.Fcollumns; z++) {
									// Note that "dictionary[s]" is used because the sth entry in the dictionary is the expected output for this input ("input[s]")
									currentFilter.weights[x][y][z] = stepGradient(learningRate, layers, out, j, f, x, y, z, dictionary[s]);
								}
							}
						}	
					}
					
					//2.b) Increment all the biases for this layer:
					for(int b = 0; b< currentLayer.biases.size(); b++){
						// Note that "dictionary[s]" is used because the sth entry in the dictionary is the expected output for this input ("input[s]")
						currentLayer.biases.get(b).value = stepGradient(learningRate, layers, out, j, b, dictionary[s]);	
					}
					
					
				}
				
				// 3. Reset stored gradients				
				// 3.a) Reset all stored gradients for all layers
				for (int j= 0; j < layers.size(); j++) {
					Layer currentLayer = layers.get(j);

					// 3.a) Reset all stored gradients for all the filters for this layer ("currentLayer")
					for (int f = 0; f< currentLayer.filters.size(); f++) {
						Filter currentFilter = currentLayer.filters.get(f);

						// 3.a.i) Reset all stored gradients for each weight within this filter ("currentFilter")
						for (int x = 0; x < currentLayer.Fdepth; x++) {
							for (int y = 0; y < currentLayer.Frows; y++) {
								for (int z = 0; z < currentLayer.Fcollumns; z++) {
									currentFilter.gradientValues[x][y][z] = -1;
								}
							}
						}
					}

					//3.b) Reset all stored gradients for all the biases for this layer
					for (int b = 0; b < currentLayer.biases.size(); b++) {
						currentLayer.biases.get(b).derivative = -1;
					}

				}
			}
			
			

		}

	}

	/**
	 * This function opens a single file (indicated by the filename passed in as
	 * a string), and feeds that file as input into the first layer of the
	 * network
	 * 
	 * @param layers
	 *            The layers that make up this network
	 * @param filename
	 *            The filename of the file to be opened/used, in string form.
	 *            Note that the file must be in same location as Align class
	 *            file.
	 * @throws IOException
	 *             Thrown if there is a problem locating or opening the file.
	 */
	public static void openFileInput(LinkedList<Layer> layers, String filename) throws IOException {
		// The file named here must be in same location as Align class file
		BufferedImage image = ImageIO.read(Align.class.getResource(filename));

		// This is necessary to use any of the OpenCV functions
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// First, convert the image to an OpenCV Mat.
		// This function can be found in image.Align.java
		Mat test = Align.bufferedImageToMat(image);

		// Next, resize the image to the size needed.
		Mat resizedImage = new Mat();
		Size sz = new Size(layers.get(0).collumns, layers.get(0).rows);
		Imgproc.resize(test, resizedImage, sz);

		// Split the image into desired number of color channels
		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here
													// in the order RGB
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
					layers.get(0).cells[c][d][e].value = temp[(d * layers.get(0).rows) + e];
				}

			}

		}	
	}
	
	public static void openFileInputBW(LinkedList<Layer> layers, String filename){
		
		URL location = FSONNetwork.class.getProtectionDomain().getCodeSource().getLocation();
		
		String urlString = location.toString();
		
		String substr = urlString.substring(5, (urlString.length() - 4));
		
		String absPath = substr.concat(filename);
		// This is necessary to use any of the OpenCV functions
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat img = Highgui.imread(absPath, 0);
		
		if (img.cols() == 0) {
		     System.out.println("Error reading file");
		}
		
		// Next, resize the image to the size needed.
		Mat resizedImage = new Mat();
		Size sz = new Size(layers.get(0).collumns, layers.get(0).rows);
		Imgproc.resize(img, resizedImage, sz);
		
		// ...and then into the cells of the first layer of the network.
		for (int d = 0; d < layers.get(0).rows; d++) {
			for (int e = 0; e < layers.get(0).collumns; e++) {
				layers.get(0).cells[0][d][e].value = (255-resizedImage.get(d, e)[0]);
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

		for (int i = 0; i < (layers.size() - 1); i++) {
			Layer currentLayer = layers.get(i);
			Layer nextLayer = layers.get(i + 1);

			switch (currentLayer.type) {
			case CONV:
				currentLayer.convolution(currentLayer.cells, currentLayer.filters, nextLayer.cells, currentLayer.step,
						currentLayer.pad, currentLayer.biases, store);
				break;
			case FULLY:
				currentLayer.full(currentLayer.cells, currentLayer.filters, nextLayer.cells[0][0], currentLayer.step,
						currentLayer.pad, currentLayer.biases,store);
				break;
			case LOCAL:
				currentLayer.local(currentLayer.cells, currentLayer.filters, nextLayer.cells, currentLayer.step,
						currentLayer.pad, currentLayer.biases, store);
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

		Layer lastLayer = layers.getLast();
		lastLayer.full(lastLayer.cells, lastLayer.filters, out, lastLayer.step, lastLayer.pad, lastLayer.biases, store);

	}

}
