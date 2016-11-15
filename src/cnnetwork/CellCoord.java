package cnnetwork;

/**
 * A class to hold the coordinates of a cell within a layer
 */
public class CellCoord {
	public int depth;	// Depth coordinate in the layer's "cells" ([x][][])
	public int row;	// Row coordinate in the layer's "cells" ([][x][])
	public int column; // Column coordinate in the layer's "cells" ([][][x])
	
	public CellCoord(int depth, int row, int column) {
		this.depth = depth;
		this.row = row;
		this.column = column;
	}
}
