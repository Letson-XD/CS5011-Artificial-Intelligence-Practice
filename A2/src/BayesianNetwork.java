
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;

public class BayesianNetwork {
    private final LinkedHashMap<String, Variable> networkNodes; // Mapping of variable names to their corresponding variable objects.

    public BayesianNetwork() {
        networkNodes = new LinkedHashMap<>();
    }

    public void addVariable(String name, ArrayList<String> outcomes) {
        networkNodes.put(name, new Variable(name, outcomes));
    }

    public Variable getVariable(String name) {
        return networkNodes.get(name);
    }

    public LinkedHashSet<String> getVariableNames() {
        return new LinkedHashSet<>(networkNodes.keySet());
    }

    public LinkedHashSet<Variable> getVariables() {
        return new LinkedHashSet<>(networkNodes.values());
    }

    public LinkedList<Factor> getFactors() {
        return new LinkedList<>(networkNodes.values().stream().map(var -> var.getFactor()).toList());
    }
}
