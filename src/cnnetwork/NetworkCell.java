package cnnetwork;

import java.util.LinkedList;

/**
 * This class defines the datatype of a single
 * cell in a layer. These cells are stored in the layer's
 * "cells" array.
 *
 */
public class NetworkCell {
	public double value; //Actual numerical value of this cell
	public int indexOfFilter; //Index in the list of filters for the previous layer that is applied to generate this cell.
	public int indexOfBias;//Index in the list of biases for the previous layer that is applied to generate this cell.
	public LinkedList<NetworkCell> nextCells; //Cells that depend on this one for input
	public LinkedList<NetworkCell> prevCells; //Cells that are used as input to this one
	
	/**
	 * Full constructor for a NetworkCell.
	 * 
	 * @param value
	 *            The actual numerical value of this cell
	 * @param indexOfFilter
	 *            The index in the list of filters for the previous layer that
	 *            is applied to generate this cell.
	 * @param indexOfBias
	 *            The index in the list of biases for the previous layer that is
	 *            applied to generate this cell.
	 * @param nextCells
	 *            Any cells in another layer that depend on this one for input.
	 * @param prevCells
	 *            Any cells in another layer that are used as input to this
	 *            cell.
	 */
	public NetworkCell(double value, int indexOfFilter, int indexOfBias, LinkedList<NetworkCell> nextCells,
			LinkedList<NetworkCell> prevCells) {
		this.value = value;
		this.indexOfFilter = indexOfFilter;
		this.indexOfBias = indexOfBias;
		this.nextCells = nextCells;
		this.prevCells = prevCells;
	}
	
	/**
	 * Default constructor for a NetworkCell. Gives an initial value of 0,
	 * indexes of -1 (to indicate no filters or biases are yet assigned), and
	 * initializes the linked lists (nextCells and prevCells).
	 */
	public NetworkCell() {
		this.value = 0.0;
		this.indexOfFilter = -1;
		this.indexOfBias = -1;
		LinkedList<NetworkCell> next = new LinkedList<NetworkCell>();
		LinkedList<NetworkCell> prev = new LinkedList<NetworkCell>();
		this.nextCells = next;
		this.prevCells = prev;
	}
}
