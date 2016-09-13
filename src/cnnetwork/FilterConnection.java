package cnnetwork;

public class FilterConnection {
	public int biasIndex;//The bias used in this computation
	public CellCoord inStart;//The top left coord of the section of cells used for input
	public CellCoord out;//The coordinates of the cell used for output

	public FilterConnection(int biasIndex, CellCoord inStart, CellCoord out) {
		this.biasIndex = biasIndex;
		this.inStart = inStart;
		this.out = out;
	}
}
