import java.util.LinkedList;

/**
 * A Node represents an agent to be utilised.
 */
public class Node {
    private Coord state;
    private Node parentNode;
    private String action;
    private double pathCost;
    private double hCost;
    private double fCost;
    private int depth;

    public Node(Coord state) {
        this.state = state;
    }

    /**GETTER METHODS */
    public Coord getState() {
        return state;
    }

    public Node getParentNode() {
        return parentNode;
    }

    public String getAction() {
        return action;
    }

    public double getFCost() {
        return fCost;
    }

    public double getHCost() {
        return hCost;
    }

    public double getPathCost() {
        return pathCost;
    }

    public int getDepth() {
        return depth;
    }

    /** SETTER METHODS */

    public void setState(Coord coord) {
        this.state = coord;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public void setParentNode(Node parentNode) {
        this.parentNode = parentNode;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public void setPathCost(Double pathCost) {
        this.pathCost = pathCost;
    }

    public void setFCost(double fCost) {
        this.fCost = fCost;
    }

    public void setHCost(double hCost) {
        this.hCost = hCost;
    }

    /**
     * Comparison method Used to establish a priority for for the informed search methods.
     * 
     * Priority ordering is as follows
     * 1. The FCost.
     * 2. The Direction of the movement.
     * 3. The Depth of the node from the start.
     */
    public int compareTo(Node n) {
        LinkedList<String> hierarchy = new LinkedList<String>(){};
        hierarchy.add(0, "Right");
        hierarchy.add(1,"Down");
        hierarchy.add(2,"Left");
        hierarchy.add(3,"Up");

        return (this.getFCost() == n.getFCost()) ?
        //
         ((hierarchy.indexOf(this.getAction()) == hierarchy.indexOf(n.getAction())) ?
        //
          ((this.getDepth() >= n.getDepth()) ? 1 : -1) :
        //
           (hierarchy.indexOf(this.getAction()) < hierarchy.indexOf(n.getAction())) ? -1 : 1) :
        
            ((this.getFCost() < n.getFCost()) ? -1 : 1);
    }
}
