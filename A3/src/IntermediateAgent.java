import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.logicng.datastructures.Tristate;
import org.logicng.formulas.Formula;
import org.logicng.formulas.FormulaFactory;
import org.logicng.io.parsers.ParserException;
import org.logicng.io.parsers.PropositionalParser;
import org.logicng.solvers.MiniSat;
import org.logicng.solvers.SATSolver;

public class IntermediateAgent extends BeginnerAgent {

    /**
     * 
     * Intermediate Agent using SAT as DFN and SPS.
     * 
     */
    public IntermediateAgent() {
        super();
    }

    /**
     * Runs the agent.
     */
    public void run() {
		// Gets the next unknown cell from the knowledge base
		// Used to catch when agent not terminated and prevent infinite loops
		Cell nextCell = kb.getNext();

        while (nextCell != null) {
            
            // If cell is 0,0 or middle cell we know it is safe so we can probe it immediately
			if ((nextCell.getX() == 0 & nextCell.getY() == 0) || (nextCell.getX() == kb.getMiddle() & nextCell.getY() == kb.getMiddle()))  {
				probeCell(nextCell); // Probes this cell
				nextCell = kb.getNext(); // Gets the next cell
				continue; // Repeats while loop
			}
            String knowledgeBase = buildKBU();

            for (Cell curr : getFrontier()) {                
                // If knowledge base is empty then finish
                if (knowledgeBase.length() == 0) {A3main.checkComplete(kb, numberOfFlags);}
                
                // Find if the cell is safe, if so probe it
                isCellSafe(knowledgeBase, curr);
    
                // Find if the cell has a tornado, if so mark it
                isCellTornado(knowledgeBase, curr);
            }


			// Get the next cell from the knowledge base
			nextCell = kb.getNext();
		}
        A3main.checkComplete(kb, numberOfFlags);
    }

    /**
     * Builds the knowledge base in Disjunctive Normal Form.
     * @return The logic sentence based on the current frontier.
     */
    public String buildKBU() {
        LinkedList<String> kbu = new LinkedList<>();

		for (Cell uncoveredCell : kb.getNonZeroUncoveredCells()) {
			// Get a list of the cellID's of the neighbours
			Set<String> coveredNeighboursID = new LinkedHashSet<>();

            kb.getUnknownNeighbours(uncoveredCell).forEach(unkN -> coveredNeighboursID.add(getID(unkN)));

            int numberOfRemainingFlags = Character.getNumericValue(uncoveredCell.getCharacter()) - kb.getFlaggedNeighbours(uncoveredCell).size();
            if (numberOfRemainingFlags > 0) {                
                LinkedList<LinkedList<String>> possibilities = encodeNeighbors(coveredNeighboursID, numberOfRemainingFlags);
                LinkedList<String> literal = new LinkedList<>();
                for (LinkedList<String> possibility : possibilities) { 
                    literal.add("(" + String.join(" & ", possibility) + ")");
                }
                kbu.add(String.join(" | ", literal));
            } else {
                SPS(false, uncoveredCell);
            }
        }
        String knowledge = "";

        for (String logic : kbu) {
            knowledge += "(" + logic + ") & ";
        }
        return knowledge;
    }

    /**
     * Processes the given neighbours into DNF based on the number of remaining flags required in the neighbourhood.
     *  
     * @param neighbours The neighbours to process.
     * @param numberOfRemainingFlags The number of remaining flags for a given neighbourhood.
     * @return All the possibilities of a tornado for the given neighbourhood.
     */
    public static LinkedList<LinkedList<String>> encodeNeighbors(Set<String> neighbours, int numberOfRemainingFlags) {
        LinkedList<LinkedList<String>> dnf = new LinkedList<>();
        if (numberOfRemainingFlags == 0) {
            LinkedList<String> conjunction = new LinkedList<>();
            for (String neighbor : neighbours) {
                conjunction.add(neighbor);
            }
            dnf.add(conjunction);
            return dnf;
        } else if (numberOfRemainingFlags == neighbours.size()) {
            LinkedList<String> conjunction = new LinkedList<>();
            for (String neighbor : neighbours) {
                conjunction.add(neighbor);
            }
            dnf.add(conjunction);
            return dnf;
        } else {
            for (Set<String> subset : getSubsets(neighbours, numberOfRemainingFlags)) {
                LinkedList<String> conjunction = new LinkedList<>();
                for (String neighbor : neighbours) {
                    if (subset.contains(neighbor)) {
                        conjunction.add(neighbor);
                    } else {
                        conjunction.add("~" + neighbor);
                    }
                }
                dnf.add(conjunction);
            }
            return dnf;
        }
    }

    /**
     * Gets all the possible combinations of the given set.
     * 
     * @param set The set to be grouped into its subsets.
     * @param k The number of combinations in each subset.
     * @return All the possible combinations of the given set as subsets.
     */
    public static Set<Set<String>> getSubsets(Set<String> set, int k) {
        Set<Set<String>> subsets = new HashSet<>();
        if (k == 0) {
            subsets.add(new HashSet<>());
            return subsets;
        } else if (k == set.size()) {
            subsets.add(set);
            return subsets;
        } else {
            String element = set.iterator().next();
            Set<String> rest = new HashSet<>(set);
            rest.remove(element);
            Set<Set<String>> subsetsWithoutElement = getSubsets(rest, k);
            Set<Set<String>> subsetsWithElement = getSubsets(rest, k - 1);
            for (Set<String> subset : subsetsWithoutElement) {
                subsets.add(subset);
            }
            for (Set<String> subset : subsetsWithElement) {
                Set<String> subsetWithElement = new HashSet<>(subset);
                subsetWithElement.add(element);
                subsets.add(subsetWithElement);
            }
            return subsets;
        }
    }

    /**
     * Gets all the unknown cells that border known cells.
     * 
     * @return A list of all the frontier unknown cells.
     */
    protected LinkedList<Cell> getFrontier() {
        Set<Cell> set = new LinkedHashSet<Cell>();
        List<LinkedList<Cell>> temp = kb.getProbed().stream().map(cell -> kb.getUnknownNeighbours(cell)).filter(list -> list.size() > 0).toList();

        for (LinkedList<Cell> cells : temp) {
            for (Cell cell : cells) {
                set.add(cell);
            }
        }
        return new LinkedList<>(set);
    }

    /**
     * Gets the unique ID for the given cell.
     * @param cell The cell to be processed.
     * @return The unique ID of the given cell.
     */
	public String getID(Cell cell) {
		return Integer.toString(((cell.getX() + 1) * 1000) + (cell.getY() + 1) * 10);
	}

	/**
	 * Checks if the knowledge base entails that the cell is safe.
     * 
	 * @param kb Knowledge base.
	 * @param cell Cell to test.
	 * @return True or false if the cell is safe to be uncovered.
	 */
	protected boolean isCellSafe(String kb, Cell cell) {

		// Gets the ID of the cell
		String cellID = getID(cell);

		// Sets the  proposition string
		String entailSafe = kb + cellID;

		// Uses entailment to check if the cell is safe
		boolean isSatisfiedSafe = false;
		try {
			isSatisfiedSafe = isSATSatisfied(entailSafe);
		} catch (ParserException e) {
			e.printStackTrace();
		}

		// If the cell is entailed to be safe, probe the cell
		if (isSatisfiedSafe) {
			probeCell(cell);
		}

		return !isSatisfiedSafe;
	}

	/**
	 * Checks if the knowledge base entails that the cell contains a tornado.
	 * @param kb Knowledge base.
	 * @param cell Cell to check.
	 * @return True or false if the cell contains a tornado.
	 */
	protected boolean isCellTornado(String kb, Cell cell) {

		// Gets the ID of the cell
		String cellID = getID(cell);

		// Sets the  proposition string
		String entailmentTornado = kb + "~" + cellID;

		// Uses entailment to check if the cell contains a tornado
		boolean isSatisfiedTornado = false;
		try {
			isSatisfiedTornado = isSATSatisfied(entailmentTornado);
		} catch (ParserException e) {
			e.printStackTrace();
		}

		// If the cell contains a tornado, mark it
		if (isSatisfiedTornado) {
			markCell(cell);
		}

		return !isSatisfiedTornado;
	}

	/**
	 * Solve a given string using the SAT solver.
     * 
	 * @param str String to solve
	 * @return True or false if the sentence is satisfiable
	 * @throws ParserException
	 */
	public boolean isSATSatisfied(String str) throws ParserException {

		FormulaFactory f = new FormulaFactory();
		PropositionalParser p = new PropositionalParser(f);
		Formula formula = p.parse(str);
		SATSolver miniSat = MiniSat.miniSat(f);
		miniSat.add(formula);
		Tristate result = miniSat.sat();
		return result.toString().equals("FALSE");
	}
}
