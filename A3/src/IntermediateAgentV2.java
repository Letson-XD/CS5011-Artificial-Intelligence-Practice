import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Set;

import org.sat4j.core.VecInt;
import org.sat4j.minisat.SolverFactory;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.IProblem;
import org.sat4j.specs.ISolver;
import org.sat4j.specs.TimeoutException;

public class IntermediateAgentV2 extends IntermediateAgent {

    /**
     * Intermediate Agent Version 2.
     * 
     * Uses SPS and SAT as DIMACS built from CFN.
     */
    public IntermediateAgentV2() {
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

            String knowledgeBase = buildCNFKBU();
            if (!knowledgeBase.isEmpty()) {                
                LinkedList<int[]> clauses = convertToDIMACS(knowledgeBase);
                for (Cell curr : getFrontier()) {                
                    // If knowledge base is empty then finish
                    
                    // Finds if the cell is safe, if so probe it
                    Boolean isTornado = false;
                    Boolean isSafe = false;
                    try {
                        isTornado = checkCell(clauses, Integer.parseInt(getID(curr)));
                        isSafe = checkCell(clauses, 0 - Integer.parseInt(getID(curr))); //Check Safe
                    } catch (NumberFormatException | TimeoutException e) {
                        e.printStackTrace();
                    } //Checks Tornado

                    if (isTornado && isSafe) {
                    } else if (isTornado) {
                        markCell(curr);
                    } else if (isSafe) {
                        probeCell(curr);
                    }
                }
            }
			// // Gets the next cell from the knowledge base
			nextCell = kb.getNext();
		}
        A3main.checkComplete(kb, numberOfFlags);
    }

    /**
     * Converts the CNF logic sentence to DIMACS.
     * 
     * @param knowledgeBase The CNF string to transform.
     * @return A list containing all the clauses as integer arrays.
     */
    private LinkedList<int[]> convertToDIMACS(String knowledgeBase) {
        String[] clauses = knowledgeBase.split("&");
        LinkedList<int[]> finalClauses = new LinkedList<>();
        for (int i = 0; i < clauses.length; i++) {
            String temp = clauses[i];
            temp = temp.replace(")", "");
            temp = temp.replace("(", "").trim();
            String[] split = temp.split("[|]");

            int[] clause = new int[split.length];
            for (int j = 0; j < split.length; j++) {
                if (!split[j].equals("|")) {
                    if (split[j].contains("~")) {
                        clause[j] = 0 - Integer.parseInt(split[j].trim().substring(1, split[j].trim().length()));
                    } else {
                        clause[j] = Integer.parseInt(split[j].trim());
                    }
                }
            }
            finalClauses.add(clause);
        }

        return finalClauses;
    }

    /**
     * Checks the given cell ID is satisfactory. 
     * @param clauses The knowledge base in DIMACS format.
     * @param cellID The cell ID to check.
     * @return Whether the knowledge base is satisfactory with the given cell ID.
     * @throws TimeoutException
     */
    private Boolean checkCell(LinkedList<int[]> clauses, int cellID) throws TimeoutException {
        ISolver solver = SolverFactory.newDefault();
        solver.newVar(10000); //Set random number.
        solver.setExpectedNumberOfClauses(clauses.size() + 1);
        int[] c = new int[1];
        c[0] = cellID;
        clauses.add(c);
        for (int[] clause : clauses) {
            try {
                solver.addClause(new VecInt(clause));
            } catch (ContradictionException e) {
                clauses.remove(c);
                return false;
            }
        }
        clauses.remove(c);
        IProblem problem = solver;

        if (problem.isSatisfiable()) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Builds the knowledge base in Conjunctive Normal Form.
     * @return The logic sentence based on the current frontier.
     */
    private String buildCNFKBU() {
        LinkedList<String> kbu = new LinkedList<>();

		for (Cell uncoveredCell : kb.getNonZeroUncoveredCells()) {
			// Gets a list of the cellID's of the neighbours
			Set<String> coveredNeighboursID = new LinkedHashSet<>();

            kb.getUnknownNeighbours(uncoveredCell).forEach(unkN -> coveredNeighboursID.add(getID(unkN)));

            int numberOfRemainingFlags = Character.getNumericValue(uncoveredCell.getCharacter()) - kb.getFlaggedNeighbours(uncoveredCell).size();
            if (numberOfRemainingFlags > 0) {
                String potentialSafe = atMost(coveredNeighboursID, kb.getUnknownNeighbours(uncoveredCell).size() - numberOfRemainingFlags, true);
                String potentialTornadoes = atMost(coveredNeighboursID, numberOfRemainingFlags, false);
                kbu.add(potentialTornadoes);
                kbu.add(potentialSafe);
            } else {
                SPS(false, uncoveredCell);
            }
        }
        kbu.removeIf(str -> str.length() == 0);
        return String.join(" & ", kbu);
    }

    /**
     * Gets all the possibilities of at most k tornados/safe cells within the given unknown neighbourhood.
     * @param unknownNeighbours The unknown neighbourhood.
     * @param k The number of at most tornados/safe cells.
     * @param safe Whether we're checking tornado cells or safe spaces.
     * @return The logic statement of at most k tornados/safe cells within the given unknown neighbourhood in CNF.
     */
    private String atMost(Set<String> unknownNeighbours, int k, Boolean safe) {
        if (k > 0) {
            LinkedList<String> strlst = new LinkedList<>();
            Set<Set<String>> encodedNeighbours = getSubsets(unknownNeighbours, k);
            LinkedList<LinkedList<String>> convert = new LinkedList<>();
            for (Set<String> set : encodedNeighbours) {
                LinkedList<String> temp = new LinkedList<String>();
                if (safe) {
                    for (String str : set) {
                        temp.add("~" + str);
                    }
                } else {
                    for (String str : set) {
                        temp.add(str);
                    }
                }
                if (encodedNeighbours.size() > 1) {
                    LinkedList<String> newTemp = new LinkedList<String>();
                    newTemp.add(String.join(" | ", temp));
                    temp = newTemp;
                }
                convert.add(temp);
            }
            convert.forEach(set -> strlst.add(String.join(" | ", set)));

            if (strlst.get(0).contains("|")) {
                return "(" + String.join(") & (", strlst) + ")" ;
            }
            return "(" + String.join(" | ", strlst) + ")" ;
        } else {
            return "";
        }
    }
}
