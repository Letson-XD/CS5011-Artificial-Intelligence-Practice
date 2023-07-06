
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;

/**
 * The Factor class contains a table to store a given CPT.
 * 
 * The table can be augmented through the joining to a different table or the summing out of a variable.
 * 
 * Other functionality includes the normalisation and assignment. 
 * Normalisation sums all the rows and divides each row by the total. Therefore, normalising each row to one.
 * Assignment eliminates rows that do not contain the given value for the given variable.
 */
public class Factor {
    private LinkedList<LinkedList<String>> table;

    public Factor() {
        this.table = new LinkedList<>();
    }

    /**
     * Eliminates the given variable from the table through a search.
     * The index of the variable is noted and used to eliminate from each row.
     * @param var The variable to be summed out.
     * @return A new factor with the given variable summed out.
     */
    public Factor sumOut(String var) {
        int eliminationIndex = table.get(0).indexOf(var);
        Factor newFactor = new Factor();
        LinkedList<String> header = new LinkedList<>();
        HashMap<String, ArrayList<Integer>> remainingValues = new HashMap<>();

        for (LinkedList<String> row : table) {
            newFactor.getTable().add(new LinkedList<>(row));
        }
        if (newFactor.getTable().get(0).size() > 2) {            
            for (int i = 0; i < newFactor.getTable().size(); i++) {
                LinkedList<LinkedList<String>> newTable = new LinkedList<LinkedList<String>>(newFactor.getTable());
                newTable.get(i).remove(eliminationIndex);
                if (i == 0) {
                    header.addAll(newTable.get(0));
                } else {
                    newTable.get(i).removeFirst();
                    String temp = String.join(" ", newTable.get(i));
                    if (remainingValues.containsKey(temp)) {
                        remainingValues.get(temp).add(i);
                    } else {
                        remainingValues.put(temp, new ArrayList<>(Arrays.asList(i)));
                    }
                }
            }
            newFactor.getTable().clear();
            for (Entry<String, ArrayList<Integer>> sumGroup : remainingValues.entrySet()) {
                Double sum = sumGroup.getValue().stream().mapToDouble(index -> Double.parseDouble(table.get(index).get(0))).reduce(0.0, (a,b) -> a + b);
                LinkedList<String> tempRow = new LinkedList<String>(Arrays.asList(sum.toString()));
                tempRow.addAll(Arrays.asList(sumGroup.getKey().split(" ")));
                newFactor.getTable().add(tempRow);
            }
            newFactor.getTable().addFirst(header);
        }
        return newFactor;
    }

    /**
     * Eliminates rows that do not contain the given value for the given variable from the table.
     * @param assignment An array containing a variable and a value as [Variable: value]
     */
    public void assign(String[] assignment) {
        int variableIndex = table.get(0).indexOf(assignment[0]);

        if (variableIndex != -1) {           
            LinkedList<LinkedList<String>> newTable = new LinkedList<LinkedList<String>>();
            for (LinkedList<String> row : table) {
                if (table.indexOf(row) != 0) {                
                    if (row.get(variableIndex).equals(assignment[1])) {
                        newTable.add(row);
                    }
                } else {
                    newTable.add(row);
                }
            }
            table = newTable;
        }
    }

    /**
     * Sums all the rows and divides each row by the total sum.
     * Therefore, normalising each row to one.
     * @return A new factor with the adjusted values.
     */
    public Factor normalize(Boolean testing) {
        Double sumOfValues = 0.0;
        Factor newFactor = new Factor();
        LinkedList<LinkedList<String>> newTable = newFactor.getTable();
        if (testing) {
            System.out.println("Before Normaisation: " + table);
        }
    
        for (LinkedList<String> row : table) {
            if (row != table.getFirst()) {
                sumOfValues += Double.parseDouble(row.get(0));
            }
            newTable.add(new LinkedList<>(row));
        }
    
        for (LinkedList<String> row : newTable) {
            if (row != newTable.getFirst()) {
                Double normalizedValue = Double.parseDouble(row.poll()) / sumOfValues;
                row.addFirst(normalizedValue.toString());
            }
        }
        if (testing) {
            System.out.println("After Normaisation: " + newFactor.getTable());
        }

        return newFactor;
    }

    /**
     * Joins two factors together.
     * Iterates through each factor's rows and finds each connecting variables.
     * Determines if the connecting variable's values are the same.
     * If they are the same, the probabilities of the rows are multiplied and a new row is created.
     * @param factor The factor to join.
     * @return A new factor with the two factors joined.
     */
    public Factor join(Factor factor) {
        if (!table.isEmpty()) {
            Factor newFactor = new Factor();
            LinkedHashSet<String> parents = new LinkedHashSet<>();
            LinkedList<LinkedList<String>> joiningTable = factor.getTable();
            LinkedList<String> header = new LinkedList<>();

            parents.addAll(joiningTable.get(0));
            for (String col : table.get(0)) {
                if (!parents.contains(col)) {
                    parents.add(col);
                }
            }
            parents.remove("prob");

            List<String> connectingNode = parents.stream().filter(str -> joiningTable.get(0).contains(str) && table.get(0).contains(str)).toList();

            if (!connectingNode.isEmpty()) {
                List<Integer> connectingIndexMain = connectingNode.stream().map(exemptNode -> table.get(0).indexOf(exemptNode)).toList();
                for (LinkedList<String> rowTable : table) {
                    if (table.indexOf(rowTable) != 0) {                    
                        for (LinkedList<String> rowJoining : joiningTable) {
                            if (joiningTable.indexOf(rowJoining) != 0) {                                
                                Boolean isCorrelated = true;
                                for (String node : connectingNode) {
                                    if (!rowTable.get(table.get(0).indexOf(node)).equals(rowJoining.get(joiningTable.get(0).indexOf(node)))) {
                                        isCorrelated = false;
                                    }
                                }

                                if (isCorrelated) {                                    
                                    LinkedList<String> temp = new LinkedList<String>();
                                    LinkedList<String> rowTableTemp = new LinkedList<>(rowTable);
                                    LinkedList<String> rowJoiningTemp = new LinkedList<>(rowJoining);

                                    Double total =  Double.parseDouble(rowTableTemp.pop()) * Double.parseDouble(rowJoiningTemp.pop());
                                    temp.addAll(rowJoiningTemp);
                                    for (int i = 0; i < rowTableTemp.size(); i++) {
                                        if (!connectingIndexMain.contains(i+1)) {
                                            temp.add(rowTableTemp.get(i));
                                        }
                                    }
                                    temp.addFirst(total.toString());

                                    newFactor.getTable().add(temp);
                                }
                            }
                        }
                    }
                }
            } else {
            }
            header.addAll(parents);
            header.addFirst("prob");
            newFactor.getTable().addFirst(header);

            return newFactor;
        } else {
            return factor;
        }
    }

    /**
     * Checks if the given variable is within the factor.
     * Checks the column headers
     * @param var The variable to check.
     * @return Whether the variable is within the table.
     */
    public Boolean contains(String var) {
        return table.get(0).contains(var);
    }

    /**
     * Retrieves the probability value for the provided variable.
     * @param assignment An array containing a variable and a value as [Variable: value]
     * @return The probability value for the provided assignment.
     */
    public Double evaluate(List<String> assignment) {
        String variable = assignment.get(0);
        String value = assignment.get(1);
        int variableIndex = table.get(0).indexOf(variable);
        Double result = 0.0;
        for (LinkedList<String> row : table) {
            if (row.get(variableIndex).equals(value)) {
                result = Double.parseDouble(row.get(0));
            }
        }
        return result;
    }

    /**
     * Retrieves the table variable.
     * @return The table variable.
     */
    public LinkedList<LinkedList<String>> getTable() {
        return table;
    }
}
