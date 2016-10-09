package cnnetwork;

import java.util.ArrayList;
import java.util.Collections;

/**
 * This class is a small utility designed to generate an array list of integers
 * in a random order.
 * 
 * Based of of code found here:
 * http://stackoverflow.com/questions/8115722/generating-unique-random-numbers-
 * in-java
 * 
 * @author Andrew Thompson
 *         (http://stackoverflow.com/users/418556/andrew-thompson), 
 *         and 
 *         Paula Rudy
 *         (https://github.com/PaulaRudy)
 *
 */

public class UniqueRandomNumbers {

	/**
	 * Returns an array list of integers in a random order in the range of 0 to
	 * "range". Use list.get(i) to access, using a "for" loop to increment i to
	 * access each in turn. Please note that the random "seed" is given by
	 * system time in milliseconds, and for a particular seed value, the
	 * 'random' instance will return the exact same sequence of pseudo random
	 * numbers.
	 * 
	 * @param range
	 *            The range (from 0 to "range") of numbers to generate.
	 *            Therefore, this also determines the size if the returned array
	 *            list.
	 * @return An array list of integers in a random order in the range of 0 to
	 *         "range".
	 */
	public static ArrayList<Integer> getRandomSet(int range) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i < range; i++) {
			list.add(new Integer(i));
		}
		Collections.shuffle(list);
		return list;

		// For your benefit, this is a sample implementation to access each
		// number:
		//	 for (int i=0; i<3; i++) {
		//		list.get(i));
		//	 }
    }
}