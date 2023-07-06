public class BasicAgent extends Agent{

	/**
	 * The Basic Agent.
	 */
    public BasicAgent() {
        super();
    }

    /**
     * Runs the agent.
     * Agent probes the cell in order for left to right and top to bottom.
     */
    public void run() {
        Cell nextCell = kb.getNext();
        while (kb.getUnknowns().size() != A3main.numberOfTornadoes) {

            if (A3main.verbose) {
                A3main.printBoard(kb.boardString());
            }

            probeCell(nextCell);
            nextCell = kb.getNext();
        }
        A3main.endGame(kb.boardString(), "\nResult: Agent alive: all solved\n");
    }
}
