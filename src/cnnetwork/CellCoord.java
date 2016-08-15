package cnnetwork;

public class CellCoord {
	int layerNum;//Index of the layer to which this cell belongs
	int depth;	// Depth coordinate in the layer's "cells" ([x][][])
	int row;	// Row coordinate in the layer's "cells" ([][x][])
	int column; // Column coordinate in the layer's "cells" ([][][x])
	
	public CellCoord(int layerNum, int depth, int row, int column) {
		this.layerNum = layerNum;
		this.depth = depth;
		this.row = row;
		this.column = column;
	}
}
