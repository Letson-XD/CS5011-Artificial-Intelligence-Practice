
import java.util.ArrayList;

public class Variable {

    private String name; // Outcomes of the variable.
    private ArrayList<String> outcomes; // Outcomes of the variable.
    private ArrayList<String> parents; // Parent variables of this variable in the BN.
    private ArrayList<String> children; // Parent variables of this variable in the BN.
    private Factor factor; // Probability table associated with this variable (linked to parents).

    public Variable(String name, ArrayList<String> outcomes) {
        this.name = name;
        this.outcomes = outcomes;
        this.parents = new ArrayList<>();
        this.children = new ArrayList<>();
    }

    public void addParent(String parent) {
        this.parents.add(parent);
    }

    public void addChild(String child) {
        this.children.add(child);
    }

    // GETTERS
    public ArrayList<String> getOutcomes() {
        return outcomes;
    }

    public Factor getFactor() {
        return factor;
    }

    public String getName() {
        return name;
    }

    public ArrayList<String> getParents() {
        return parents;
    }

    public ArrayList<String> getChildren() {
        return children;
    }

    // SETTERS
    public void setFactor(Factor probTable) {
        this.factor = probTable;
    }
}
