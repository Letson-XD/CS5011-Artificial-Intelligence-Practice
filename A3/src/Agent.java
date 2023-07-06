public class Agent {

    KnowledgeBase kb;
	int numberOfFlags;

	/**
	 * The agent functionality.
	 * 
	 * Based on the lecture material.
	 * 
	 */
    public Agent() {
        kb = new KnowledgeBase(A3main.board);
    }

	/**
	 * Probes the given cell for it's character.
	 * 
	 * IF character is '0': Probe the neighbours.
	 * IF character is 't': Game over.
	 * 
	 * @param cell The cell to be probed.
	 */
    public void probeCell(Cell cell) {
		updateCell(cell);
		A3main.checkSuccess(kb, numberOfFlags);
		
        switch (A3main.board[cell.getX()][cell.getY()]) {
            case '0' -> probeNeighbours(cell);
            case 't' -> A3main.endGame(kb.boardString(), "Result: Agent dead: found mine");
        }
    }

	/**
	 * Probes the unknown neighbours of the given cell.
	 * 
	 * @param cell The cell to get the neighbours for.
	 */
    public void probeNeighbours(Cell cell) {
		kb.getUnknownNeighbours(cell).forEach(neighbour -> probeCell(neighbour));
	}

	/**
	 * Marks the unknown neighbours of the given cell.
	 * 
	 * @param cell The cell to get the neighbours for.
	 */
    public void markNeighbours(Cell cell) {
		kb.getUnknownNeighbours(cell).forEach(neighbour -> markCell(neighbour));
	}

	/**
	 * Updates a given cell with the character from the real board.
	 * Moves the cell from order to probed list in the Knowledge base.
	 * 
	 * @param cell Cell to update.
	 */
    public void updateCell(Cell cell) {
		cell.setCharacter(A3main.board[cell.getX()][cell.getY()]);
		kb.getOrder().remove(cell);
		kb.getProbed().add(cell);
    }

	/**
	 * Marks a given cell with the '*' character.
	 * Increments the number of flags by one.
	 * Checks if the game is won.
	 * 
	 * @param cell Cell to mark.
	 */
	public void markCell(Cell cell) {
		cell.setCharacter('*');
		numberOfFlags++;
		A3main.checkSuccess(kb, numberOfFlags);
	}

	/**
	 * Checks that the number of neighbours already marked equals the clue in the
	 * given cell.
	 * @param  cell Cell to check
	 * @return If clue = number of marked neighbours
	 */
	public boolean allFreeNeighbours(Cell cell) {

		// Gets cell value as an integer
		int cellVal = Character.getNumericValue(cell.getCharacter());

		// Gets the number of marked adjacent cells
		int c = numberOfCInNeighbourhood(cell, '*');
		return cellVal == c;
	}

	/**
	 * Gets the number of neighbours containing the character c.
	 * @param cell The cell to get the neighbours for.
	 * @param c The character to check.
	 * @return The number of neighbours containing the character.
	 */
	public int numberOfCInNeighbourhood(Cell cell, char c) {
		if (c == '*') {
			return kb.getFlaggedNeighbours(cell).size();
		} else if (c == '?') {
			return kb.getUnknownNeighbours(cell).size();
		} else {
			return kb.getKnownNeighbours(cell).stream().filter(temp -> temp.getCharacter() == c).toList().size();
		}
    }

    /**
	 * Checks that the number of unmarked and covered neighbour cell equals the
	 * cell clue subtract the number of already marked neighbour cells.
	 * @param  cell Cell to check
	 * @return      True or false if #unmarked&covered cells = clue - #marked cells
	 */
	public boolean allMarkedNeighbours(Cell cell) {

		// Gets cell value as an integer
		int cellVal = Character.getNumericValue(cell.getCharacter());

		// Gets the number of marked adjacent cells
		int c = numberOfCInNeighbourhood(cell, '*');

		// Gets the number of unknown adjacent cells
		int q = numberOfCInNeighbourhood(cell, '?');

		return q == cellVal - c;
	}
}
