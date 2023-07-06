public class BeginnerAgent extends Agent {

	/**
	 * Beginner Agent.
	 */
    public BeginnerAgent() {
        super();
    }

	/**
	 * Runs the Agent.
	 * 
	 * For each cell in the board.
	 * Gets the uncovered neighbours and gets SPS.
	 * Loops through until game is won or stuck.
	 */
    public void run() {

		Cell nextCell = kb.getNext();
		Cell repeatCell = null;

		while (nextCell != null) {
			if ((nextCell.getX() == 0 & nextCell.getY() == 0) | (nextCell.getX() == kb.getMiddle() & nextCell.getY() == kb.getMiddle()))  {
			    probeCell(nextCell);
			} else {
                Boolean edited = false;
                // Check every uncovered adjacent neighbour of nextCell
			    for (Cell temp : kb.getKnownNeighbours(nextCell)) {

				    edited = SPS(edited, temp);
			    }
                if (!edited) {
                    kb.addToBack(nextCell);
					if (repeatCell == null) {
						repeatCell = nextCell;
                	} else if (repeatCell == nextCell) {
						A3main.checkComplete(kb, numberOfFlags);
					}
				} else {
					repeatCell = null;
				}
            }

			// Gets the next cell from the knowledge base
			nextCell = kb.getNext();
		}
        A3main.checkComplete(kb, numberOfFlags);
    }

	/**
	 * Single Point Strategy method.
	 * 
	 * Checks All Free Neighbours and All Marked Neighbours.
	 * 
	 * @param edited Whether the previous cell caused an edit to the board.
	 * @param cell The cell to be checked.
	 * @return Whether the given cell caused an edit to the board.
	 */
	public Boolean SPS(Boolean edited, Cell cell) {
		// If #clue = #dangers marked...
		if (allFreeNeighbours(cell) == true) {
		    probeNeighbours(cell);	// Probes all neighbouring cells
		    edited = true;
		// If #?cells = #clue - #marked
		} else if (allMarkedNeighbours(cell) == true) {
		    markNeighbours(cell);		// Marks all neighbouring cells
		    edited = true;
		}
		return edited;
	}
}
