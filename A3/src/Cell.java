
import java.awt.Point;

public class Cell {
    private Point coordinate;
	private char character;

	Cell(int x, int y) {
		this.coordinate = new Point(x,y);
		character = '?';
	}

	/**
	 * Setter for character.
	 * @param character
	 */
	public void setCharacter(char character) {
		this.character = character;
	}

	/**
	 * Getter for character.
	 * @return Character.
	 */
	public char getCharacter() {
		return character;
	}

	/**
	 * Getter for x coord.
	 * @return The x coord.
	 */
	public int getX() {
		return (int) coordinate.getX();
	}

	/**
	 * Setter for x coord.
	 * @param x The x to set.
	 */
	public void setX(int x) {
		coordinate.setLocation(x, coordinate.getY());
	}

	/**
	 * Getter for y coord.
	 * @return The y coord.
	 */
	public int getY() {
		return (int) coordinate.getY();
	}

	/**
	 * Setter for y coord.
	 * @param y The y to set.
	 */
	public void setY(int y) {
		coordinate.setLocation(coordinate.getX(), y);
	}

	/**
	 * Returns the string that contains the information about this coordinate.
	 * @return Coordinate information string
	 */
    @Override
	public String toString() {
		return coordinate.toString();
	}
}
