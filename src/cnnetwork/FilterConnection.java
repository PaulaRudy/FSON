package cnnetwork;

public class FilterConnection {
	public int biasIndex;//The bias used in this computation
	public CellCoord inStart;//The top left coord of the section of cells used for input
	public CellCoord out;//The coordinates of the cell used for output
	double gradientValue;//The value of the gradient at this location. 
						//Used during backpropagation to store previously computed values.
						//This will always be with respect to the appropriate weight because 
						//weights are calculated layer by layer and are overwritten before 
						//being used.
	
	public FilterConnection(int biasIndex, CellCoord inStart, CellCoord out, double gradientValue) {
		this.biasIndex = biasIndex;
		this.inStart = inStart;
		this.out = out;
		this.gradientValue = gradientValue;
	}
}
