package cnnetwork;

/**
 * This class stores the definition of a cell within a layer. Cells are given a
 * place to store a calculated derivative value to aid in backpropagation.
 * 
 */
public class Cell {
	public double derivative;//Used to store the value of the derivative at this location. A value of -1 here indicates a derivative has not been found yet.
	public double value;// the actual value of the cell.
	
	public Cell(double derivative, double value) {
		this.derivative = derivative;
		this.value = value;
	}
	
	public Cell() {
		this.derivative = -1;// A value of -1 indicates no derivative has been found yet
		this.value = 0.0;
	}
	
	public Cell(double value) {
		this.derivative = -1;// A value of -1 indicates no derivative has been found yet
		this.value = value;
	}
	
}
