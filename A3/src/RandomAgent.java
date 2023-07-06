public class RandomAgent extends IntermediateAgent {

	/**
	 * Random Agent
	 * 
	 * Based on a real Mine Sweeper strategy.
	 * 
	 * Builds on the SPS implementation to try and find a solution instead of giving up.
	 */
    public RandomAgent() {
        super();
        probeCell(kb.getNext());
    }

	/**
	 * Runs the agent.
	 */
    public void run() {
        if (!attemptSPS(getFrontier().getFirst())) {
            A3main.printBoard(kb.boardString());
            probeCell(kb.getRandom());
            run();
        } else {
            A3main.checkComplete(kb, numberOfFlags);
        }
    }

	/**
	 * Same SPS method as the one used in Beginner only configured for recursive programming.
	 * @param cell The next cell.
	 * @return Whether the SPS achieved a success. If false, perform random probing and run again.
	 */
    public boolean attemptSPS(Cell cell) {

		 // Gets the next unknown cell from the knowledge base
		Cell nextCell = cell;
		Cell repeatCell = null;

		// For every unknown cell in the knowledge base...
		while (nextCell != null) {
			if ((nextCell.getX() == 0 & nextCell.getY() == 0) | (nextCell.getX() == kb.getMiddle() & nextCell.getY() == kb.getMiddle()))  {
			    probeCell(nextCell);
			} else {
                Boolean edited = false;
                // Checks every uncovered adjacent neighbour of nextCell
			    for (Cell temp : kb.getKnownNeighbours(nextCell)) {

				    edited = SPS(edited, temp);
			    }
                if (!edited) {
                    kb.addToBack(nextCell);
					if (repeatCell == null) {
						repeatCell = nextCell;
                	} else if (repeatCell == nextCell) {
						return false;
					}
				} else {
					repeatCell = null;
				}
            }
			// Gets the next cell from the knowledge base
			nextCell = kb.getNext();
		}
        return true;
    }
}
