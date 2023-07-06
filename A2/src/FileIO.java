
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

public class FileIO {

    /**
     * Reads the given XML file to the Bayesian network instance.
     * Loops through all the given tags to retrieve data for the instance.
     * @param file The XML file to parse.
     * @return A complete Bayesian network.
     */
    public static BayesianNetwork readFile(File file) {
        BayesianNetwork bayesianNetwork = new BayesianNetwork();

        try {
            DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder documentBuilder = documentBuilderFactory.newDocumentBuilder();
            Document doc = documentBuilder.parse(file);
            doc.getDocumentElement().normalize();

            NodeList variables = doc.getElementsByTagName("VARIABLE");
            NodeList definitions = doc.getElementsByTagName("DEFINITION");

            for (int i = 0; i < variables.getLength(); i++) {
                Element variable = (Element) variables.item(i);
                // Get variable name.
                String varName = variable.getElementsByTagName("NAME").item(0).getTextContent();
                // Get variable outcomes.
                ArrayList<String> varOutcomes = new ArrayList<>();
                NodeList outcomes = variable.getElementsByTagName("OUTCOME");
                for (int currentOutcome = 0; currentOutcome < outcomes.getLength(); currentOutcome++) {
                    varOutcomes.add(outcomes.item(currentOutcome).getTextContent());
                }

                // Create Variable object and add to the BN.
                bayesianNetwork.addVariable(varName, varOutcomes);
            }

            // Loop over definitions in the XML file to get the parents of each variable and associated probabilities.
            for (int currentDefinition = 0; currentDefinition < definitions.getLength(); currentDefinition++) {
                Element currentElement = (Element) definitions.item(currentDefinition);
                // Get variable this definition is for.
                String variableForStr = currentElement.getElementsByTagName("FOR").item(0).getTextContent();
                Variable variableFor = bayesianNetwork.getVariable(variableForStr);
                // Get probability table and add probability table to the variable.
                String probTableString = currentElement.getElementsByTagName("TABLE").item(0).getTextContent();
                NodeList parents = currentElement.getElementsByTagName("GIVEN");

                // Adds the children and parents to each variable.
                for (int currentParent = 0; currentParent < parents.getLength(); currentParent++) {
                    bayesianNetwork.getVariable(parents.item(currentParent).getTextContent()).addChild(variableFor.getName());
                    variableFor.addParent(parents.item(currentParent).getTextContent());
                }

                ArrayList<String> probTable = new ArrayList<>(Arrays.asList(probTableString.split(" ")));
                Factor factor = new Factor();
                LinkedList<String> headers = new LinkedList<>(Arrays.asList("prob", variableFor.getName()));
                headers.addAll(variableFor.getParents());
                factor.getTable().add(headers);
                ArrayList<LinkedList<String>> variableList = new ArrayList<LinkedList<String>>();

                //Builds the Factor/Table.
                if (variableFor.getParents().size() == 2) {
                    for (String parentAOutcomes : bayesianNetwork.getVariable(variableFor.getParents().get(0)).getOutcomes()) {
                        for (String parentBOutcomes : bayesianNetwork.getVariable(variableFor.getParents().get(1)).getOutcomes()) {
                            for (String childOutcomes : variableFor.getOutcomes()) {
                                variableList.add(new LinkedList<>(Arrays.asList(childOutcomes, parentAOutcomes, parentBOutcomes)));
                            }
                        }
                    }
                } else if (variableFor.getParents().size() == 1) {
                    for (String parentOutcomes : bayesianNetwork.getVariable(variableFor.getParents().get(0)).getOutcomes()) {
                        for (String childOutcomes : variableFor.getOutcomes()) {
                            variableList.add(new LinkedList<>(Arrays.asList(childOutcomes, parentOutcomes)));
                        }
                    }
                } else {
                    for (String childOutcomes : variableFor.getOutcomes()) {
                        variableList.add(new LinkedList<>(Arrays.asList(childOutcomes)));
                    }
                }
                for (int i = 0; i < probTable.size(); i++) {
                    variableList.get(i).addFirst(probTable.get(i));
                }
                for (LinkedList<String> row : variableList) {
                    factor.getTable().add(row);
                }
                variableFor.setFactor(factor);
                // Get variables this variable is affected by and add variableGivens as parents to this variableFor.
            }
        } catch (Exception e) {
            System.out.println("readBNFromFile() Exception - Could not parse Bayesian network from file: " + file.getName());
            System.exit(-1);
        }
        return bayesianNetwork;
    }
}
