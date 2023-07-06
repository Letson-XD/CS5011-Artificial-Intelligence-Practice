import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Scanner;
import java.io.*;

/**
 * Starter Code provided by lf28.
 * All methods and functions have been updated as the system developed.
 */

public class A2main {
	static BayesianNetwork bayesianNetwork = new BayesianNetwork();
	static File file = new File("");
	static LinkedList<String> query = new LinkedList<>();

	/**
	 * Start point of the system.
	 * Produces a probability based on a given query and Bayesian network.
	 * @param args <P1|P2|P3> <BNA.xml|BNB.xml|BNC.xml|BNOne.xml|BNCycle.xml|BNNegative.xml> <True/False>
	 */
	public static void main(String[] args) {
		String variable = "";
		String value = "";
		Boolean testing = false;
		LinkedList<String> eliminationOrder = new LinkedList<>();
		LinkedList<String[]> evidence = new LinkedList<>();
		if (args.length < 2) {
			System.out.println("java A2main <P1|P2|P3> <BNA.xml|BNB.xml|BNC.xml|BNOne.xml|BNCycle.xml|BNNegative.xml> <Test>");
			return;
		} else if (args.length == 3) {
			testing = Boolean.parseBoolean(args[2]);
		}
		try {
			file = new File(args[1]);
			verifyBN(file);
		} catch (Exception e) {
			System.out.println(e);
			System.exit(-1);
		}
		Scanner sc = new Scanner(System.in);

		try {
			query = getQueriedNode(sc);
			if (args[0].equals("P2")) {
				eliminationOrder = getOrder(sc);
			} else if (args[0].equals("P3")) {
				evidence = getEvidence(sc);
			}
			variable = query.get(0);
			value = query.get(1);
		} catch (Exception e) {
			System.out.println(e);
			System.exit(-1);
		}

		switch (args[0]) {
			case "P1" -> printResult(new VariableElimination(bayesianNetwork).query(variable, value, testing));
			case "P2" -> printResult(new VariableElimination(bayesianNetwork, eliminationOrder).query(variable, value, testing));
			case "P3" -> {
				VariableElimination va = new VariableElimination(bayesianNetwork);
				va.addEvidence(evidence);
				printResult(va.query(variable, value, testing));
			}
		}

		sc.close();
	}

	/**
	 * Tests the given Bayesian network against three tests.
	 * Test One: If the network is not a DAG.
	 * Test Two: If the network contains any negatives.
	 * Test Three: If the network does not sum to one.
	 * @param file The Bayesian network to be tested and used for variable elimination.
	 */
	private static void verifyBN(File file) throws IOException {
		bayesianNetwork = FileIO.readFile(file);
		verifyDAG();
		verifyCPTs(bayesianNetwork.getFactors());
	}

	/**
	 * Verifies the Bayesian network is a DAG.
	 * That the system does not contain any loops.
	 * @throws IOException If the Bayesian network does contain a loop.
	 */
	private static void verifyDAG() throws IOException {
		ArrayList<Variable> roots = new ArrayList<Variable>(bayesianNetwork.getVariables().stream().filter(variable -> variable.getParents().isEmpty()).toList());
		if (roots.size() != 0) {
			for (Variable root : roots) {
				LinkedList<Variable> currentVariable = new LinkedList<>(Arrays.asList(root));
				HashSet<String> expanded = new HashSet<>();
				while (!currentVariable.isEmpty()) {
					Variable variable = currentVariable.pop();
					expanded.add(variable.getName());
	
					for (String child : variable.getChildren()) {
						if (!expanded.contains(child)) {
							currentVariable.add(bayesianNetwork.getVariable(child));
						} else {
							throw new IOException("Invalid Bayesian Network: There is a cycle between " + child + " and " + variable.getName());
						}
					}
				}
			}
		} else {
			throw new IOException("Invalid Bayesian Network: There are no root variables");
		}
	}

	/**
	 * Verifies the CPTs contain no negative values.
	 * Verifies the CPTs sum to one.
	 * @param factors All the factors within the Bayesian network.
	 * @throws IOException If the system contains any negative values or if the values do not sum to one.
	 */
	private static void verifyCPTs(LinkedList<Factor> factors) throws IOException {
		for (Factor factor : factors) {
			String tableName = factor.getTable().get(0).get(1);
			Double sum = 0.0;
			for (LinkedList<String> row : factor.getTable()) {
				if (!factor.getTable().getFirst().equals(row)) {
					Double value = Double.parseDouble(row.get(0));
					sum += value;
					if (0 > value) {
						throw new IOException("Invalid Bayesian Network: Negative probabilities present in table " + tableName);
					}
					if (factor.getTable().indexOf(row) % bayesianNetwork.getVariable(tableName).getOutcomes().size() == 0) {
						if (sum != 1.0) {
							throw new IOException("Invalid Bayesian Network: Probabilities do not sum to one in table " + tableName);
						} else {
							sum = 0.0;
						}
					}
				}
			}
		}
	}

	/**
	 * Obtains the queried evidence from the user.
	 * @param sc The scanner to read.
	 * @return A list containing the requested evidence to be applied to the Bayesian network.
	 * @throws IOException If the provided variable does not exist within the Bayesian network.
	 */
	private static LinkedList<String[]> getEvidence(Scanner sc) throws IOException {
		System.out.println("Evidence:");
		LinkedList<String[]> evidenceSet = new LinkedList<String[]>();
		String[] line = sc.nextLine().split(" ");

		for (String st : line) {
			String[] ev = st.split(":");
			evidenceSet.add(ev);
		}
		for (String[] evidence  : evidenceSet) {
			if (bayesianNetwork.getVariable(evidence[0]) == null) {
				throw new IOException("Evidence Error: Variable " + evidence[0] + " does not exist within this Bayesian Network.");
			}
		}
		return evidenceSet;
	}

	/**
	 * Obtains the queried order from the user.
	 * @param sc The scanner to read.
	 * @return A list containing the requested order for the variables.
	 * @throws IOException If the provided variable does not exist within the Bayesian network.
	 */
	private static LinkedList<String> getOrder(Scanner sc) throws IOException {
		System.out.println("Order:");
		String[] vals = sc.nextLine().split(",");
		for (String val  : vals) {
			if (bayesianNetwork.getVariable(val) == null) {
				throw new IOException("Elimination Order Error: Variable " + val + " does not exist within this Bayesian Network.");
			}
			if (query.get(0).equals(val)) {
				throw new IOException("Elimination Order Error: Variable " + val + " already exists as the query.");
			}
		}
		return new LinkedList<>(Arrays.asList(vals));
	}

	/**
	 * Obtains the queried node from the user.
	 * @param sc The scanner to read.
	 * @return A list containing the the requested variable with a given value.
	 * @throws IOException If the provided variable does not exist within the Bayesian network.
	 */
	private static LinkedList<String> getQueriedNode(Scanner sc) throws IOException {
		System.out.println("Query:");
		String[] val = sc.nextLine().split(":");
		if (bayesianNetwork.getVariable(val[0]) == null) {
			throw new IOException("Query Error: Variable " + val[0] + " does not exist within this Bayesian Network.");
		}

		return new LinkedList<>(Arrays.asList(val));
	}

	/**
	 * Formats and prints the result to 5 decimal places.
	 * @param result The result to format and print.
	 */
	private static void printResult(double result) {
		DecimalFormat dd = new DecimalFormat("#0.00000");
		System.out.println(dd.format(result));
	}
}
