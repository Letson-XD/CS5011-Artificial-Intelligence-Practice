import java.util.HashSet;
import java.util.LinkedList;

/**
 * The informed search is used for the A* search and the Best-First search algorithms.
 */
public class InformedSearch extends Search {

    public InformedSearch(String algo, Map map, Coord start, Coord goal) {
        super(algo, map, start, goal);
    }

    /**
     * The Node is built similarly to the uninformed verison. However, the hcost and fcost and utilised.
     * @param parentNode The parent that created this node.
     * @param childState The coordinate of the child node.
     * @return A new node.
     */
    private Node makeNode(Node parentNode, Coord childState) {
        Node n = new Node(childState);
        n.setParentNode(parentNode);
        if (n.getParentNode() != null) {
            n.setAction(getAction(parentNode.getState(), childState));
            n.setDepth(parentNode.getDepth() + 1);
            n.setPathCost(parentNode.getPathCost() + 1);
            n.setHCost(calcHCost(childState));
            n.setFCost(n.getHCost());
            if (algo.equals("AStar")) {
                n.setFCost(n.getHCost() + n.getPathCost());
            }

        } else {
            n.setHCost(calcHCost(childState));
            n.setFCost(n.getHCost());
            n.setPathCost(0.0);
        }
        return n;
    }

    /**
     * Calculates the hcost based on the Manhattan Distance.
     * @param state The coordinate to be used in the calculation.
     * @return The new hcost value.
     */
    private double calcHCost(Coord state) {
        int[] coordinates = calculateManhattanDistance(state.getR(), state.getC());
        return Math.abs(coordinates[0] - goalCost[0]) + Math.abs(coordinates[1] - goalCost[1]) + Math.abs(coordinates[2] - goalCost[2]);
    }

    /**
     * The general search algorithm tailored the informed search.
     */
    public void search() {
        int numberOfNodes = 0;
        Node n = makeNode(null, start);
        LinkedList<Node> frontier = new LinkedList<Node>();

        //Adds the inital state to the frontier.
        frontier.add(n);
        System.out.println(frontier.stream().map(node -> node.getState().toString() + ":" + node.getFCost()).toList().toString().replace(" ", ""));
        
        HashSet<Node> explored = new HashSet<>();

        while (!frontier.isEmpty()) {
            //Gets next based on search algorithm
            n = getNext(frontier);
            numberOfNodes++;
            //Tests if the current state is the goal state
            if (testGoal(n.getState(), goal)) {
                buildResult(true, n, numberOfNodes);
                return;
            }
            explored.add(n);
            Coord[] actions = breakTie(n.getState(), map);

            //Gets the children of the current node
            for (Coord action : actions) {
                Node child = makeNode(n, action);
                if (checkExpanded(explored, child.getState()) && checkFrontier(frontier, child.getState())) {
                    frontier = getPriority(frontier, child);
                }
            }
            if (!frontier.isEmpty()) {
                System.out.println(frontier.stream().map(node -> node.getState().toString() + ":" + node.getFCost()).toList().toString().replace(" ", ""));
            }
        }
        buildResult(false, n, numberOfNodes);
    }

    /**
     * Adds the child node and reorganises the frontier based on the priority.
     * @param frontier The frontier to be reorganised
     * @param child The child to be added 
     * @return A newly organised frontier.
     */
    public LinkedList<Node> getPriority(LinkedList<Node> frontier, Node child) {
        frontier.add(child);
        frontier.sort(Node::compareTo);
        return frontier;
    }
}

