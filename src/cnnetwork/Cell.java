package cnnetwork;

/**
 * This class stores the definition of a cell within a layer. Cells are given a
 * place to store a calculated derivative value to aid in backpropagation.
 * 
 */
public class Cell {
	public double derivative;//Used to store the value of the derivative at this location. A value of Double.NaN here indicates a derivative has not been found yet.
	public double value;// the actual value of the cell.
	public double previousValue;// Used during backpropagation to avoid using newly incremented values during learning.
	
	public Cell(double derivative, double value) {
		this.derivative = derivative;
		this.value = value;
	}
	
	public Cell() {
		this.derivative = Double.NaN;// A value of Double.NaN indicates no derivative has been found yet
		this.value = 0.0;
	}
	
	public Cell(double value) {
		this.derivative = Double.NaN;// A value of Double.NaN indicates no derivative has been found yet
		this.value = value;
	}
	
}
