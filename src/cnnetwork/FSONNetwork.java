package cnnetwork;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
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
		this.layers.get(0).convolution(this.layers.get(0).cells, this.layers.get(0).filters, this.layers.get(1).cells, this.layers.get(0).step, this.layers.get(0).pad, this.layers.get(0).biases);
		this.layers.get(1).pool(this.layers.get(1).cells, this.layers.get(1).filters, this.layers.get(2).cells, this.layers.get(1).step, this.layers.get(1).Fcollumns);
		this.layers.get(2).convolution(this.layers.get(2).cells, this.layers.get(2).filters, this.layers.get(3).cells, this.layers.get(2).step, this.layers.get(2).pad, this.layers.get(2).biases);
		this.layers.get(3).pool(this.layers.get(3).cells, this.layers.get(3).filters, this.layers.get(4).cells, this.layers.get(3).step, this.layers.get(3).Fcollumns);
		this.layers.get(4).convolution(this.layers.get(4).cells, this.layers.get(4).filters, this.layers.get(5).cells, this.layers.get(4).step, this.layers.get(4).pad, this.layers.get(4).biases);
		this.layers.get(5).local(this.layers.get(5).cells, this.layers.get(5).filters, this.layers.get(6).cells, this.layers.get(5).step, this.layers.get(5).pad, this.layers.get(5).biases);
		this.layers.get(6).full(this.layers.get(6).cells, this.layers.get(6).filters, this.layers.get(7).cells[0][0], this.layers.get(6).step, this.layers.get(6).pad, this.layers.get(6).biases);
		this.layers.get(7).full(this.layers.get(7).cells, this.layers.get(7).filters, this.out, this.layers.get(7).step, this.layers.get(7).pad, this.layers.get(7).biases);
		Layer.softmax(this.out);
	}

	/**
	 * This function computes the partial derivative of the total error with
	 * respect to a given weight within a network. This is a recursive function
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
	 * This is a helper function to the function with the call signature:
	 * 
	 * computePartialDerivative(LinkedList<Layer> layers, Cell[] out, int
	 * layerIndex, int filterIndex, int depth, int row, int column, double[]
	 * expected)
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
	 *            c
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

}
