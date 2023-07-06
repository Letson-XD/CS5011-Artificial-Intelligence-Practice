
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

public class KnowledgeBase {
    private int width; // Width of board
    private int length; // Length of board
    private Cell[][] board; // The agents board.
    private LinkedList<Cell> order = new LinkedList<Cell>(); // The cells to be probed.
    private LinkedList<Cell> probed = new LinkedList<Cell>(); // The cells that are already probed.

    /**
     * Constructor for KnowledgeBase.
     * 
     * @param board Original board
     */
    public KnowledgeBase(char[][] board) {
        length = board.length;
        width = board[0].length;
        setUpKnowledgeBase(board);
    }
  
    /**
     * Sets up the knowledge base by creating every unblocked cell on the board.
     * 
     * @param board Original board.
     */
    private void setUpKnowledgeBase(char[][] board) {
        this.board = new Cell[length][width];

        for (int i = 0; i < length; i++) {
            for (int j = 0; j < width; j++) {
                Cell cell = new Cell(i, j);
                order.add(cell);
                this.board[i][j] = cell;
            }
        }
    }

    /**
     * Gets the next cell in the order cells list.
     * 
     * @return Next cell
     */
    public Cell getNext() {
        if (order.isEmpty()) {
            return null;
        } else {
            Cell temp = order.pop();
            return temp;
        }
    }

    /**
     * Adds the given cell to the back of the order queue.
     * 
     * @return Next cell
     */
    public void addToBack(Cell cell) {
        probed.remove(cell);
        order.addLast(cell);
    }

    /**
     * Gets all the neighbours for the given cell.
     * 
     * If it's a square board, it will get the additional neighbours.
     * 
     * @param x The x coordinate for the cell.
     * @param y The y coordinate for the cell.
     * @return A list of the neighbourhood.
     * @throws IndexOutOfBoundsException
     */
    private ArrayList<Cell> getNeighbours(int x, int y) throws IndexOutOfBoundsException {
        if (x < 0 || x > width || y < 0 || y > length) {
            throw new IndexOutOfBoundsException();
        } else {
            ArrayList<Cell> neighbors = new ArrayList<Cell>();
            if (A3main.isSquare) {
                neighbors.add(new Cell(x+1, y-1));
                neighbors.add(new Cell(x-1, y+1));
            }
            neighbors.add(new Cell(x-1, y-1)); // Get Left
            neighbors.add(new Cell(x-1, y)); // Get Right
            neighbors.add(new Cell(x, y-1)); // Get Up Left
            neighbors.add(new Cell(x, y+1)); // Get Up Right
            neighbors.add(new Cell(x+1, y)); // Get Down Left
            neighbors.add(new Cell(x+1, y+1)); // Get Down Right

            neighbors = new ArrayList<>(neighbors
                                .stream()
                                .filter(cell -> cell.getX() >= 0 && cell.getX() < width && cell.getY() < length && cell.getY() >= 0)
                                .map(newCell -> board[newCell.getX()][newCell.getY()])
                                .toList());
            return neighbors;
        }
    }

    /**
     * Gets all the unknown neighbours of the given cell.
     * @param cell The given cell.
     * @return A list of all the unknown neighbours.
     */
    public LinkedList<Cell> getUnknownNeighbours(Cell cell) {
        return new LinkedList<>(getNeighbours(cell.getX(), cell.getY()).stream().filter(co -> co.getCharacter() == '?').toList());
    }

    /**
     * Gets all the flagged neighbours of the given cell.
     * @param cell The given cell.
     * @return A list of all the flagged neighbours.
     */
    public LinkedList<Cell> getFlaggedNeighbours(Cell cell) {
        return new LinkedList<>(getNeighbours(cell.getX(), cell.getY()).stream().filter(co -> co.getCharacter() == '*').toList());
    }

    /**
     * Gets all the unknown neighbours of the given cell.
     * @param cell The given cell.
     * @return A list of all the unknown neighbours.
     */
    public LinkedList<Cell> getKnownNeighbours(Cell cell) {
        return new LinkedList<>(getNeighbours(cell.getX(), cell.getY()).stream().filter(co -> co.getCharacter() != '?' && co.getCharacter() != '*').toList());
    }

    /**
     * Gets the list of all non-zero uncovered cells.
     * @return  List of all non-zero uncovered cells.
     */
    public LinkedList<Cell> getNonZeroUncoveredCells() {
        LinkedList<Cell> temp = new LinkedList<Cell>();
        
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < width; j++) {
                if (board[i][j].getCharacter() != '?' && board[i][j].getCharacter() != '0' && board[i][j].getCharacter() != '*') {
                    temp.add(board[i][j]);
                }
            }
        }
        return temp;
    }

    /**
     * Gets all the known cells on the board.
     * @return A list of all the unknown cells.
     */
    public LinkedList<Cell> getUnknowns() {
        LinkedList<Cell> temp = new LinkedList<Cell>();

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board.length; j++) {
                if (board[i][j].getCharacter() == '?') {
                    temp.add(board[i][j]);
                }
            }
        }
        return temp;
    }

    /**
     * Moves a cell from the start to the end of the unknownCells list.
     * Used when a given cell cannot be marked or uncovered yet so is added to the
     * back of the 'queue'.
     * 
     * @param cell Cell to move.
     */
    public void moveCellToEndOfQueue(Cell cell) {
        order.addLast(order.pop());
    }

    /**
     * Retrieves the board.
     * 
     * @return The board.
     */
    public Cell[][] getBoard() {
        return board;
    }

    /**
     * Gets the length of the board.
     * 
     * @return The length of the board.
     */
    public int getLength() {
        return length;
    }

    /**
     * Gets the width of the board.
     * 
     * @return The width of the board.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Gets the coordinate at the middle of the board.
     * 
     * @return the x/y coordinate of the middle of the board.
     */
    public int getMiddle() {
        return getLength()/2;
    }

    /**
     * Gets the probed list.
     * 
     * @return The probed list.
     */
    public LinkedList<Cell> getProbed() {
        return probed;
    }

    /**
     * Gets the order list.
     * 
     * @return The order list.
     */
    public LinkedList<Cell> getOrder() {
        return order;
    }

    /**
     * Converts the cell array to a character array.
     * 
     * @return
     */
    public char[][] boardString() {
        char[][] board = new char[length][width];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (this.board[i][j].getCharacter() == 't') {
                    board[i][j] = '-';
                } else {
                    board[i][j] = this.board[i][j].getCharacter();
                }
            }
        }
        return board;
    }

    /**
     * Gets a random unknown cell.
     * 
     * @return The random cell.
     */
    public Cell getRandom() {
        Random rand = new Random();
        Cell randomCell = getUnknowns().get(rand.nextInt(getUnknowns().size()));
        return randomCell;
    }
}
