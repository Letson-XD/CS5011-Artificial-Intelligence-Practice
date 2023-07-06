import java.util.Arrays;

/**
 * Main Route into search functionality
 */
public class A1main {

	public static void main(String[] args) {
		if (args.length < 2) {
			System.out.println("java A1main <DFS|BFS|AStar|BestF|BiDir> <ConfID> <Coord>");
			return;
		}
		Conf conf = Conf.valueOf(args[1]);

		try {
			Arrays
			.asList(args)
			.subList(2, args.length)
			.forEach(str -> conf.getMap().getMap()[Integer.parseInt(str.substring(1, 2))][Integer.parseInt(str.substring(3, 4))] = 1);
		} catch (Exception e) {
			//Not Weather Inclusive.
		}

		runSearch(args[0],conf.getMap(),conf.getS(),conf.getG());
	}

	/**
	 * Starts the correct search algorithm.
	 * @param algo The search algorithm to be used.
	 * @param map The map to be used for the search.
	 * @param start The starting coordinate
	 * @param goal The goal coordinate.
	 */
	private static void runSearch(String algo, Map map, Coord start, Coord goal) {
		switch(algo) {
			case "BFS" -> new UninformedSearch(algo, map, start, goal).search();
			case "DFS" -> new UninformedSearch(algo, map, start, goal).search();
			case "AStar" -> new InformedSearch(algo, map, start, goal).search();
			case "BestF" -> new InformedSearch(algo, map, start, goal).search();
			case "BiDir" -> new UninformedSearch(algo, map, start, goal).biDirectionalSearch();
		}
	}
}
