import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Stack;

public class UninformedSearch extends Search {

    public UninformedSearch(String algo, Map map, Coord start, Coord goal) {
        super(algo, map, start, goal);
    }

    private Node makeNode(Node parentNode, Coord childState) {
        Node n = new Node(childState);
        n.setParentNode(parentNode);
        if (n.getParentNode() != null) {
            n.setAction(getAction(parentNode.getState(), childState));
            n.setDepth(parentNode.getDepth() + 1);
            n.setPathCost(parentNode.getPathCost() + 1);
        }
        return n;
    }

    /**
     * The general search algorithm tailored the Uninformed search.
     */
    public void search() {
        int numberOfNodes = 0;
        Node n = makeNode(null, start);
        Deque<Node> frontier = new LinkedList<Node>();

        //Adds the inital state to the frontier.
        frontier.add(n);
        System.out.println(frontier.stream().map(node -> node.getState().toString()).toList().toString().replace(" ", ""));
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

            //Fix to make DFS compatible with the heirarchy specification.
            if (algo.equals("DFS")) {
                Collections.reverse(Arrays.asList(actions));
            }

            //Gets the children of the current node
            for (Coord action : actions) {
                Node child = makeNode(n, action);
                if (checkExpanded(explored, child.getState()) && checkFrontier(frontier, child.getState())) {
                    frontier.addLast(child);
                }
            }
            if (!frontier.isEmpty()) {
                System.out.println(frontier.stream().map(node -> node.getState().toString()).toList().toString().replace(" ", ""));
            }
        }
        buildResult(false, n, numberOfNodes);
    }

    /**
     * The general search algorithm tailored the Uninformed search.
     */
    public void biDirectionalSearch() {
        int numberOfNodes = 0;
        //Creates two agents. One from the start and one from the goal coordinates.
        Node startNode = makeNode(null, start);
        Node endNode = makeNode(null, goal);
        LinkedList<Node> frontierStart = new LinkedList<Node>();
        LinkedList<Node> frontierEnd = new LinkedList<Node>();

        frontierStart.add(startNode);
        frontierEnd.add(endNode);

        System.out.println("Agent Start: " + frontierStart.stream().map(node -> node.getState().toString()).toList().toString().replace(" ", ""));
        System.out.println("Agent End: " + frontierEnd.stream().map(node -> node.getState().toString()).toList().toString().replace(" ", ""));

        HashSet<Node> exploredStart = new HashSet<>();
        HashSet<Node> exploredEnd = new HashSet<>();

        while (!frontierStart.isEmpty() && !frontierEnd.isEmpty()) {
            startNode = getNext(frontierStart);
            endNode = getNext(frontierEnd);
            numberOfNodes = numberOfNodes + 2;
            
            exploredStart.add(startNode);
            exploredEnd.add(endNode);

            //Check intersect after taking from the frontier.
            if (testIntersect(exploredStart, exploredEnd) != null) {
                buildResultBiDir(true, testIntersect(exploredStart, exploredEnd), numberOfNodes);
                return;
            }

            //Gets the children for the start agent.
            for (Coord action : breakTie(startNode.getState(), map)) {
                Node child = makeNode(startNode, action);
                if (checkExpanded(exploredStart, child.getState()) && checkFrontier(frontierStart, child.getState())) {
                    frontierStart.addLast(child);
                }
            }

            //Gets the children for the end agent.
            for (Coord action : breakTie(endNode.getState(), map)) {
                Node child = makeNode(endNode, action);
                if (checkExpanded(exploredEnd, child.getState()) && checkFrontier(frontierEnd, child.getState())) {
                    frontierEnd.addLast(child);
                }
            }

            if (!frontierStart.isEmpty()) {
                System.out.println("Agent From Start: " + frontierStart.stream().map(node -> node.getState().toString()).toList().toString().replace(" ", ""));
            }
            if (!frontierEnd.isEmpty()) {
                System.out.println("Agent From End: " + frontierEnd.stream().map(node -> node.getState().toString()).toList().toString().replace(" ", ""));
            }
        }
        buildResultBiDir(false, null, numberOfNodes);
        return;
    }

    /**
     * Builds the specified return statement.
     * 
     * @param success Whether a solution is found.
     * @param intersectNodes The node that are intersecting.
     * @param numberOfNodes The number of nodes searched to find a solution.
     */
    public void buildResultBiDir(Boolean success, Node[] intersectNodes, int numberOfNodes) {
        if (success) {
            String path = "";
            String direction = "";
            Double pathCost = 0.0;
            Stack<Node> history = new Stack<>();
            LinkedList<Node> endNodeHistory = new LinkedList<Node>();
            
            
            Node intersectFromStart = intersectNodes[0];
            Node intersectFromEnd = intersectNodes[1];

            while (intersectFromEnd.getParentNode() != null) {
                endNodeHistory.addFirst(intersectFromEnd);
                intersectFromEnd = intersectFromEnd.getParentNode();
            }
            try {
                endNodeHistory.removeLast();
                history.push(intersectFromEnd);
                
            } catch (Exception e) {
                // There is only one node in list
            }

            endNodeHistory.forEach(n -> history.push(n));

            while (intersectFromStart.getParentNode() != null) {
                history.push(intersectFromStart);
                intersectFromStart = intersectFromStart.getParentNode();
            }

            path += intersectFromStart.getState();
            Node current = intersectFromStart;

            while (!history.empty()) {
                current = history.pop();
                path += current.getState();
                direction += getAction(intersectFromStart.getState(), current.getState()) + " ";
                intersectFromStart = current;
                pathCost++;
            }

            System.out.println(path + "\n" + direction + "\n" + pathCost + "\n" + numberOfNodes);
        } else {
            System.out.println("fail\n" + numberOfNodes);
        }
        return;
    }

    /**
     * Next goal test condition. Testing if there is an interestion between the explored set of both agents.
     * 
     * @param exploredStart The explored set of the start agent.
     * @param exploredEnd The explored set of the end agent.
     * @return Either the intersecting nodes or null.
     */

    private Node[] testIntersect(HashSet<Node> exploredStart, HashSet<Node> exploredEnd) {
        for (Node end : exploredEnd) {
            for (Node start : exploredStart) {
                if (start.getState().equals(end.getState())) {
                    return new Node[] {start, end};
                }
            }
        }
        return null;
    }
}
