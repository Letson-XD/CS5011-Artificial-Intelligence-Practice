import org.logicng.io.parsers.ParserException;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.TimeoutException;

public class A3main {

	static boolean verbose=false; // Prints the formulas for SAT if true.
	static boolean isSquare=false; // Reshapes the board into a square shape.
	static char[][] board; 
    static int numberOfTornadoes; // Length of board

	public static void main(String[] args) throws ParserException, TimeoutException, ContradictionException, TimeoutException {

		if (args.length>2 && args[2].equals("verbose") ){
			verbose=true; // Prints the formulas for SAT if true
		}
		if (args.length>2 && args[2].equals("square")){
			isSquare=true;
		} else if (args.length>3 && args[3].equals("square")) {
			isSquare=true;
		}

		// Read input from command line.

		// Agent type
		System.out.println("-------------------------------------------\n");
		System.out.println("Agent " + args[0] + " plays " + args[1] + "\n");

		// World
		World world = World.valueOf(args[1]);

		board = world.map;
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				if (board[i][j] == 't') {
					numberOfTornadoes++;
				}
			}
		}
		printBoard(board);
		System.out.println("Start!");

		switch (args[0]) {
		case "P1" -> new BasicAgent().run();
		case "P2" -> new BeginnerAgent().run();
		case "P3" -> new IntermediateAgent().run();
		case "P4" -> new IntermediateAgentV2().run();
		case "P5" -> new RandomAgent().run();
		}
	}

	/**
	 * Prints the end results of the game.
	 * @param board The final agents board.
	 * @param message The final message.
	 */
	public static void endGame(char[][] board, String message) {
		System.out.println("Final map");
		printBoard(board);
		System.out.print(message);
		System.exit(0);
	}

	// Prints the board in the required format - PLEASE DO NOT MODIFY.
	public static void printBoard(char[][] board) {
		System.out.println();
		// First line
		for (int l = 0; l < board.length + 5; l++) {
			System.out.print(" ");// Shift to start
		}
		for (int j = 0; j < board[0].length; j++) {
			System.out.print(j);// x indexes
			if (j < 10) {
				System.out.print(" ");
			}
		}
		System.out.println();
		// Second line.
		for (int l = 0; l < board.length + 3; l++) {
			System.out.print(" ");
		}
		for (int j = 0; j < board[0].length; j++) {
			System.out.print(" -");// Separator
		}
		System.out.println();
		// The board.
		for (int i = 0; i < board.length; i++) {
			for (int l = i; l < board.length - 1; l++) {
				System.out.print(" ");// Fill with left-hand spaces
			}
			if (i < 10) {
				System.out.print(" ");
			}

			System.out.print(i + "/ ");// Index+separator
			for (int j = 0; j < board[0].length; j++) {
				System.out.print(board[i][j] + " ");// Value on the board
			}
			System.out.println();
		}
		System.out.println();
	}

	/**
	 * Check if the agent has successfully completed the game.
	 * @param kb The agent's knowledge base.
	 * @param numberOfFlags The number of flags on the agent's board.
	 */
	public static void checkSuccess(KnowledgeBase kb, int numberOfFlags) {
		if (numberOfFlags == numberOfTornadoes) {
            for (int i = 0; i < kb.getLength(); i++) {
                for (int j = 0; j < kb.getWidth(); j++) {
                    if (kb.getBoard()[i][j].getCharacter() == '?') {
                        kb.getBoard()[i][j].setCharacter(A3main.board[i][j]);
                    } else if (kb.getBoard()[i][j].getCharacter() == 't') {
                        A3main.endGame(kb.boardString(), "Result: Fail");
                    }
                }
            }
			A3main.endGame(kb.boardString(), "Result: Agent alive: all solved");
		}
	}

	/**
	 * Checks if the game was successful.
	 * If not, checks the board for any '?' characters to see if the agent couldn't find a solution.
	 * @param kb The agent's knowledge base.
	 * @param numberOfFlags The number of flags on the agent's board.
	 */
    public static void checkComplete(KnowledgeBase kb, int numberOfFlags) {
		checkSuccess(kb, numberOfFlags);
        for (int i = 0; i < kb.getLength(); i++) {
            for (int j = 0; j < kb.getWidth(); j++) {
                if (kb.getBoard()[i][j].getCharacter() == '?') {
                    A3main.endGame(kb.boardString(), "Result: Agent not terminated");
                }
            }
        }
    }
}