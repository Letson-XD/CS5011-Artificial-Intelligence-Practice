import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

/**
 * A General search class that is used foir the informed and uninformed search algorithms.
 */
public class Search {
    Map map;
    Coord start;
    Coord goal;
    String algo;
    int[] goalCost = new int[3];

    public Search(String algo, Map map, Coord start, Coord goal) {
        this.algo = algo;
        this.map = map;
        this.start = start;
        this.goal = goal;
        this.goalCost = calculateManhattanDistance(goal.getR(), goal.getC());
    }

    /**
     * Determines the dirtection of movement from the parent state to the child.
     */
    public String getAction(Coord parentState, Coord childState) {
        if (parentState.getR() > childState.getR()) {
            return "Up";
        } else if (parentState.getR() < childState.getR()) {
            return "Down";
        } else {
            if (parentState.getC() > childState.getC()) {
                return "Left";
            } else {
                return "Right";
            }
        }
    }

    /**
     * Calculates the direction in which the triangle is pointing in (Up/Down)
     * @param row The row coordinate of the triangle.
     * @param col The column coordinate of the triangle.
     * @return A 0 if the triangle is pointing Up. 1 if triangle is pointing Down.
     */
    public int getDirection(int row, int col) {
        return ((row + col) % 2 == 0) ? 0 : 1;
    }

    /**
     * Calculates the Manhattan Distance for a single triangle.
     * @param row The row coordinate of the triangle.
     * @param col The column coordinate of the triangle.
     * @return An array of 3 values representing the manhattan distance.
     */
    public int[] calculateManhattanDistance(int row, int col) {
        int difference = row + col - getDirection(row, col);
        int a = -row;
        int b =  difference / 2;
        int c = ((difference / 2) - row) + getDirection(row, col);
    
        return new int[]{a,b,c};
    }

    /**
     * Retrieves the next node from the frontier based on the search algorithm.
     * @param frontier The frontier of the search algorithm.
     * @return The chosen Node.
     */
    public Node getNext(Deque<Node> frontier) {
        return (algo.equals("DFS")) ? frontier.pollLast() : frontier.pollFirst();
    }

    /**
     * The goal test to determine whether a solution is found.
     * @param state The current agent coordinate.
     * @param goal The goal coordinate.
     * @return Whether a solution has been found.
     */
    public boolean testGoal(Coord state, Coord goal) {
        return state.equals(goal);
    }

    /**
     * Checks the frontier to see if the current coordinate has been added already.
     * @param frontier The frontier to be checked.
     * @param coord The coordinate to be checked.
     * @return Whether the coordinate has been added already.
     */
    public boolean checkFrontier(Queue<Node> frontier, Coord coord) {
        return !frontier.stream().map(Node::getState).anyMatch(c -> coord.equals(c));
    }

    /**
     * Checks the explored set to see if the current coordinate has been added already.
     * @param explored The explored set to be checked.
     * @param coord The coordinate to be checked.
     * @return Whether the coordinate has been added already.
     */
    public boolean checkExpanded(HashSet<Node> explored, Coord coord) {
        try {
            explored.stream().filter(n -> coord.equals(n.getState())).toList().get(0);
            return false;
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * Retrieves the legal children of a given parent state. The children are added to the returned array based on the following order
     * 1. Right
     * 2. Down
     * 3. Left
     * 4. Up
     * The children are then check for legality.
     * 
     * @param state The parent of the children.
     * @param map The map to be checked.
     * @return The legal children of the parent state.
     */
    public Coord[] breakTie(Coord state, Map map) {
        int direction = getDirection(state.getR(), state.getC());
        int maxHeight = map.getMap().length - 1;
        int maxLength = map.getMap()[state.getR()].length - 1;
        LinkedList<Coord> actions = new LinkedList<Coord>();

        //Right
        actions.add(new Coord(state.getR(), state.getC() + 1));

        //Up
        if (direction == 0) {
            actions.add(new Coord(state.getR() + 1, state.getC()));
        }

        //Left
        actions.add(new Coord(state.getR(), state.getC() - 1));   

        //Down
        if (direction == 1) {
            actions.add(new Coord(state.getR() - 1, state.getC()));
        }

        //Checking if the coord is within the grid and not an island.
        actions = new LinkedList<Coord>(actions
        .stream()
        .filter(s -> s.getC() >= 0 && s.getC() <= maxLength && s.getR() >= 0 && s.getR() <= maxHeight && map.getMap()[s.getR()][s.getC()] != 1)
        .toList());

        return actions.toArray(new Coord[0]);
    }

    /**
     * Builds the specified return statement.
     * 
     * @param success Whether a solution is found.
     * @param child The node that reached the goal.
     * @param numberOfNodes The number of nodes searched to find a solution.
     */
    public void buildResult(Boolean success, Node child, int numberOfNodes) {
        if (success) {
            String path = "";
            String direction = "";
            Double pathCost = 0.0;
            Stack<Node> history = new Stack<>();
            Node n = child;
            while (n.getParentNode() != null) {
                history.push(n);
                n = n.getParentNode();
            }
            path += n.getState();
            while (!history.empty()) {
                n = history.pop();
                path += n.getState();
                direction += n.getAction() + " ";
                pathCost++;
            }
            System.out.println(path + "\n" + direction + "\n" + pathCost + "\n" + numberOfNodes);
        } else {
            System.out.println("fail\n" + numberOfNodes);
        }
        return;
    }
}
