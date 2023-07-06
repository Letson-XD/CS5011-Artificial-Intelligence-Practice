
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class VariableElimination {

    BayesianNetwork bn;
    LinkedList<String[]> evidenceSet;
    LinkedList<String> orderOfElimination;

    public VariableElimination(BayesianNetwork bn) {
        this.bn = bn;
        this.evidenceSet = new LinkedList<>();
    }

    public VariableElimination(BayesianNetwork bn, LinkedList<String> orderOfElimination) {
        this.bn = bn;
        this.orderOfElimination = orderOfElimination;
        this.evidenceSet = new LinkedList<>();
    }

    /**
     * Performs variable elimination on the provided Bayesian network.
     * 1.   Builds an order of elimination.
     * 2.   Cycles through the order of elimination.
     * 2.1. Joins all relating factor to current variable then sums out variable.
     * 2.2. Removes relating factors and adds new factor to system.
     * 
     * @param variable The queried variable.
     * @param value The queried value for the provided variable.
     * @return The probability of the value for the given variable.
     */
    public Double query(String variable, String value, Boolean testing) {
        LinkedList<String> querySet = new LinkedList<>(Arrays.asList(variable, value));
        LinkedList<Factor> factors = bn.getFactors();

        if (orderOfElimination == null) {
            orderOfElimination = getEliminationOrder(variable);
        }
        if (testing) {
            System.out.println("Order of Elimination: " + orderOfElimination);
        }
        if (!evidenceSet.isEmpty()) {
            for (String[] evidence : evidenceSet) {
                factors
                .stream()
                .filter(factor -> factor.getTable().get(0).get(1).equals(evidence[0]))
                .forEach(factor -> {
                    if (testing) {
                        System.out.println("Before assignment :" + factor.getTable());
                    }
                    factor.assign(evidence);
                    if (testing) {
                        System.out.println("After assignment :" + factor.getTable());
                    }
                });
            }
        }
        orderOfElimination.addLast(variable);

        for (int i = 0; i < orderOfElimination.size(); i++) {
            Factor eliminationFactor = new Factor();
            String node = orderOfElimination.get(i);
            List<Factor> involvingFactors = factors.stream().filter(factor -> factor.contains(node)).toList();

            for (Factor factor : involvingFactors) {
                if (testing) {
                    System.out.println("Before Joining: " + eliminationFactor.getTable());
                    System.out.println("Joining With: " + factor.getTable());
                }
                eliminationFactor = eliminationFactor.join(factor);
                if (testing) {
                    System.out.println("After Joining: " + eliminationFactor.getTable());
                }
                factors.remove(factor);
            }
            if (testing) {
                System.out.println("Before Summing out " + node + ": " + eliminationFactor.getTable());
            }
            eliminationFactor = eliminationFactor.sumOut(node);
            if (testing) {
                System.out.println("After Summing out: " + eliminationFactor.getTable());
            }
            factors.add(eliminationFactor);
        }
        return factors.pop().normalize(testing).evaluate(querySet);
    }

    /**
     * Add evidence to the necessary Factors.
     * 
     * @param evidence The evidence to add in the form of [variable, value]
     */
    public void addEvidence(LinkedList<String[]> evidence) {
        this.evidenceSet = evidence;
    }

    /**
     * Obtains the order for the variable elimination based of breadth-first search.
     * 
     * @param variableName The name of the queried variable.
     * @return The ordered list of variable to eliminate.
     */
    private LinkedList<String> getEliminationOrder(String variableName) {
		Queue<Variable> order = new LinkedList<>();
		LinkedList<String> stack = new LinkedList<>();

        
		order.add(bn.getVariable(variableName));
		do {
            Variable current = order.poll();
			order.addAll(current.getParents().stream().map(parentName -> bn.getVariable(parentName)).toList());
			stack.addAll(current.getParents().stream().filter(parent -> !stack.contains(parent)).toList());
		} while (!order.isEmpty());
		if (orderOfElimination != null && !(orderOfElimination.get(0).equals(""))) {
            orderOfElimination.forEach(node -> {
                if (stack.contains(node)) {
                    stack.remove(stack.indexOf(node));
				}
				stack.addLast(node);
			});
		} else {
            Collections.reverse(stack);
		}
        for (String variable : bn.getVariableNames()) {
            if (!stack.contains(variable) && !variable.equals(variableName)) {
                stack.addLast(variable);
            }
        }
		return stack;
	}
}
