import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.*;
import java.lang.Math;
import java.io.File;

/**
 * Represents a node with x and y coordinates and a cost.
 */
class Node {
    private int x;
    private int y;
    private int cost;

    public Node(int x, int y, int cost) {
        this.x = x;
        this.y = y;
        this.cost = cost;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getCost() {
        return cost;
    }
}

/**
 * Represents a problem instance with nodes and a distance matrix.
 */
class ProblemInstance {
    private List<Node> nodes;
    private int[][] distanceMatrix;

    public ProblemInstance() {
        nodes = new ArrayList<>();
    }

    /**
     * Reads nodes from a CSV file. Each line should have x;y;cost
     * @param filePath Path to the CSV file
     * @throws IOException If file reading fails
     */
    public void readCSV(String filePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.trim().split(";");
            if (parts.length != 3) {
                continue; // Skip invalid rows
            }
            try {
                int x = Integer.parseInt(parts[0]);
                int y = Integer.parseInt(parts[1]);
                int cost = Integer.parseInt(parts[2]);
                nodes.add(new Node(x, y, cost));
            } catch (NumberFormatException e) {
                // Skip rows with invalid integers
                continue;
            }
        }
        reader.close();
    }

    /**
     * Computes the Euclidean distance matrix between nodes, rounded to the nearest integer.
     */
    public void computeDistanceMatrix() {
        int n = nodes.size();
        distanceMatrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            Node nodeI = nodes.get(i);
            for (int j = i + 1; j < n; j++) {
                Node nodeJ = nodes.get(j);
                double dist = Math.sqrt(Math.pow(nodeI.getX() - nodeJ.getX(), 2) +
                        Math.pow(nodeI.getY() - nodeJ.getY(), 2));
                int roundedDist = (int) Math.round(dist);
                distanceMatrix[i][j] = roundedDist;
                distanceMatrix[j][i] = roundedDist;
            }
        }
    }

    public List<Node> getNodes() {
        return nodes;
    }

    public int[][] getDistanceMatrix() {
        return distanceMatrix;
    }
}

/**
 * Represents a solution with a path and its objective value.
 */
class Solution {
    private List<Integer> path;
    private int objectiveValue;

    public Solution(List<Integer> path, int objectiveValue) {
        this.path = new ArrayList<>(path);
        this.objectiveValue = objectiveValue;
    }

    public List<Integer> getPath() {
        return path;
    }

    public int getObjectiveValue() {
        return objectiveValue;
    }
}

/**
 * Abstract class for heuristic methods.
 */
abstract class Heuristic {
    protected Random random = new Random();

    /**
     * Generates a solution based on the heuristic.
     * @param instance The problem instance
     * @param k Number of nodes to select
     * @param startNode The starting node index
     * @return A Solution object
     */
    public abstract Solution generateSolution(ProblemInstance instance, int k, int startNode);
}

/**
 * Implements the Random Solution heuristic.
 */
class RandomSolution extends Heuristic {

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        List<Node> nodes = instance.getNodes();
        int n = nodes.size();
        List<Integer> selectedNodes = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            selectedNodes.add(i);
        }
        Collections.shuffle(selectedNodes, random);
        selectedNodes = selectedNodes.subList(0, Math.min(k, selectedNodes.size())); // Select k nodes randomly
        int objective = computeObjective(selectedNodes, instance);
        return new Solution(selectedNodes, objective);
    }

    /**
     * Computes the objective function: sum of path lengths + sum of node costs
     */
    private int computeObjective(List<Integer> path, ProblemInstance instance) {
        int totalDistance = 0;
        int k = path.size();
        int[][] distanceMatrix = instance.getDistanceMatrix();
        List<Node> nodes = instance.getNodes();
        for (int i = 0; i < k; i++) {
            int from = path.get(i);
            int to = path.get((i + 1) % k);
            totalDistance += distanceMatrix[from][to];
        }
        int totalCost = 0;
        for (int node : path) {
            totalCost += nodes.get(node).getCost();
        }
        return totalDistance + totalCost;
    }
}

/**
 * Implements the Nearest Neighbor heuristic by adding nodes to the end of the path.
 */
class NearestNeighborEnd extends Heuristic {

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        List<Node> nodes = instance.getNodes();
        int n = nodes.size();
        if (n == 0) {
            return new Solution(new ArrayList<>(), 0);
        }
        List<Integer> path = new ArrayList<>();
        path.add(startNode);
        Set<Integer> selected = new HashSet<>();
        selected.add(startNode);
        int[][] distanceMatrix = instance.getDistanceMatrix();

        while (path.size() < k) {
            int last = path.get(path.size() - 1);
            int nearest = -1;
            int minDist = Integer.MAX_VALUE;
            for (int node = 0; node < n; node++) {
                if (!selected.contains(node)) {
                    int dist = distanceMatrix[last][node];
                    if (dist < minDist) {
                        minDist = dist;
                        nearest = node;
                    }
                }
            }
            if (nearest != -1) {
                path.add(nearest);
                selected.add(nearest);
            } else {
                break; // No more nodes to add
            }
        }

        int objective = computeObjective(path, instance);
        return new Solution(path, objective);
    }

    /**
     * Computes the objective function: sum of path lengths + sum of node costs
     */
    private int computeObjective(List<Integer> path, ProblemInstance instance) {
        int totalDistance = 0;
        int k = path.size();
        int[][] distanceMatrix = instance.getDistanceMatrix();
        List<Node> nodes = instance.getNodes();
        for (int i = 0; i < k; i++) {
            int from = path.get(i);
            int to = path.get((i + 1) % k);
            totalDistance += distanceMatrix[from][to];
        }
        int totalCost = 0;
        for (int node : path) {
            totalCost += nodes.get(node).getCost();
        }
        return totalDistance + totalCost;
    }
}

/**
 * Implements the Nearest Neighbor heuristic by adding nodes at any position in the path.
 */
class NearestNeighborAny extends Heuristic {

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        List<Node> nodes = instance.getNodes();
        int n = nodes.size();
        if (n == 0) {
            return new Solution(new ArrayList<>(), 0);
        }
        List<Integer> path = new ArrayList<>();
        path.add(startNode);
        Set<Integer> selected = new HashSet<>();
        selected.add(startNode);
        int[][] distanceMatrix = instance.getDistanceMatrix();

        while (path.size() < k) {
            int nearest = -1;
            int minDist = Integer.MAX_VALUE;
            for (int node = 0; node < n; node++) {
                if (!selected.contains(node)) {
                    for (int p_node : path) {
                        int dist = distanceMatrix[p_node][node];
                        if (dist < minDist) {
                            minDist = dist;
                            nearest = node;
                        }
                    }
                }
            }
            if (nearest != -1) {
                // Find the best position to insert the nearest node
                int bestIncrease = Integer.MAX_VALUE;
                int bestPos = -1;
                for (int i = 0; i < path.size(); i++) {
                    int current = path.get(i);
                    int next = path.get((i + 1) % path.size());
                    int increase = distanceMatrix[current][nearest] + distanceMatrix[nearest][next] - distanceMatrix[current][next];
                    if (increase < bestIncrease) {
                        bestIncrease = increase;
                        bestPos = i + 1;
                    }
                }
                if (bestPos != -1) {
                    path.add(bestPos, nearest);
                    selected.add(nearest);
                } else {
                    // If no best position found, append at the end
                    path.add(nearest);
                    selected.add(nearest);
                }
            } else {
                break; // No more nodes to add
            }
        }

        int objective = computeObjective(path, instance);
        return new Solution(path, objective);
    }

    /**
     * Computes the objective function: sum of path lengths + sum of node costs
     */
    private int computeObjective(List<Integer> path, ProblemInstance instance) {
        int totalDistance = 0;
        int k = path.size();
        int[][] distanceMatrix = instance.getDistanceMatrix();
        List<Node> nodes = instance.getNodes();
        for (int i = 0; i < k; i++) {
            int from = path.get(i);
            int to = path.get((i + 1) % k);
            totalDistance += distanceMatrix[from][to];
        }
        int totalCost = 0;
        for (int node : path) {
            totalCost += nodes.get(node).getCost();
        }
        return totalDistance + totalCost;
    }
}

/**
 * Implements the Greedy Cycle heuristic.
 */
class GreedyCycle extends Heuristic {

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        List<Node> nodes = instance.getNodes();
        int n = nodes.size();
        if (n == 0) {
            return new Solution(new ArrayList<>(), 0);
        }
        List<Integer> path = new ArrayList<>();
        path.add(startNode);
        Set<Integer> selected = new HashSet<>();
        selected.add(startNode);
        int[][] distanceMatrix = instance.getDistanceMatrix();

        while (path.size() < k) {
            int bestNode = -1;
            int bestScore = Integer.MAX_VALUE;
            for (int node = 0; node < n; node++) {
                if (!selected.contains(node)) {
                    // Find the minimum increase in distance for insertion
                    int minIncrease = Integer.MAX_VALUE;
                    for (int i = 0; i < path.size(); i++) {
                        int current = path.get(i);
                        int next = path.get((i + 1) % path.size());
                        int increase = distanceMatrix[current][node] + distanceMatrix[node][next] - distanceMatrix[current][next];
                        if (increase < minIncrease) {
                            minIncrease = increase;
                        }
                    }
                    // Define score as increase + node cost
                    int score = minIncrease + nodes.get(node).getCost();
                    if (score < bestScore) {
                        bestScore = score;
                        bestNode = node;
                    }
                }
            }
            if (bestNode != -1) {
                // Insert the best_node at the position that minimizes the increase
                int bestIncrease = Integer.MAX_VALUE;
                int bestPos = -1;
                for (int i = 0; i < path.size(); i++) {
                    int current = path.get(i);
                    int next = path.get((i + 1) % path.size());
                    int increase = distanceMatrix[current][bestNode] + distanceMatrix[bestNode][next] - distanceMatrix[current][next];
                    if (increase < bestIncrease) {
                        bestIncrease = increase;
                        bestPos = i + 1;
                    }
                }
                if (bestPos != -1) {
                    path.add(bestPos, bestNode);
                    selected.add(bestNode);
                } else {
                    // Append at the end if no better position found
                    path.add(bestNode);
                    selected.add(bestNode);
                }
            } else {
                break; // No more nodes to add
            }
        }

        int objective = computeObjective(path, instance);
        return new Solution(path, objective);
    }

    /**
     * Computes the objective function: sum of path lengths + sum of node costs
     */
    private int computeObjective(List<Integer> path, ProblemInstance instance) {
        int totalDistance = 0;
        int k = path.size();
        int[][] distanceMatrix = instance.getDistanceMatrix();
        List<Node> nodes = instance.getNodes();
        for (int i = 0; i < k; i++) {
            int from = path.get(i);
            int to = path.get((i + 1) % k);
            totalDistance += distanceMatrix[from][to];
        }
        int totalCost = 0;
        for (int node : path) {
            totalCost += nodes.get(node).getCost();
        }
        return totalDistance + totalCost;
    }
}

/**
 * Utility class to compute statistics for a list of solutions.
 */
class Statistics {
    private int minObjective;
    private int maxObjective;
    private double avgObjective;
    private List<Integer> bestPath;

    public Statistics(int minObjective, int maxObjective, double avgObjective, List<Integer> bestPath) {
        this.minObjective = minObjective;
        this.maxObjective = maxObjective;
        this.avgObjective = avgObjective;
        this.bestPath = bestPath;
    }

    public int getMinObjective() {
        return minObjective;
    }

    public int getMaxObjective() {
        return maxObjective;
    }

    public double getAvgObjective() {
        return avgObjective;
    }

    public List<Integer> getBestPath() {
        return bestPath;
    }

    /**
     * Computes statistics from a list of solutions.
     * @param solutions List of Solution objects
     * @return A Statistics object
     */
    public static Statistics computeStatistics(List<Solution> solutions) {
        if (solutions == null || solutions.isEmpty()) {
            return new Statistics(0, 0, 0.0, new ArrayList<>());
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        long sum = 0;
        List<Integer> bestPath = new ArrayList<>();
        for (Solution sol : solutions) {
            int obj = sol.getObjectiveValue();
            if (obj < min) {
                min = obj;
                bestPath = sol.getPath();
            }
            if (obj > max) {
                max = obj;
            }
            sum += obj;
        }
        double avg = (double) sum / solutions.size();
        return new Statistics(min, max, avg, bestPath);
    }
}

/**
 * The main class to execute the program.
 */
public class Main {
    public static void main(String[] args) {
        // Define the input directory
        String inputDirPath = "inputs";
        File inputDir = new File(inputDirPath);

        if (!inputDir.exists() || !inputDir.isDirectory()) {
            System.err.println("Input directory '" + inputDirPath + "' does not exist or is not a directory.");
            return;
        }

        // List all CSV files in the input directory
        File[] inputFiles = inputDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));

        if (inputFiles == null || inputFiles.length == 0) {
            System.err.println("No CSV files found in the input directory.");
            return;
        }

        // Initialize heuristics
        List<Heuristic> heuristics = new ArrayList<>();
        heuristics.add(new RandomSolution());
        heuristics.add(new NearestNeighborEnd());
        heuristics.add(new NearestNeighborAny());
        heuristics.add(new GreedyCycle());

        // Iterate over each input file
        for (File inputFile : inputFiles) {
            String fileName = inputFile.getName();
            String instanceName = fileName.substring(0, fileName.lastIndexOf('.')); // Remove .csv extension
            String filePath = inputFile.getPath();

            System.out.println("Processing instance: " + instanceName);

            // Initialize problem instance
            ProblemInstance instance = new ProblemInstance();
            try {
                System.out.println("Reading nodes from CSV...");
                instance.readCSV(filePath);
                System.out.println("Computing distance matrix...");
                instance.computeDistanceMatrix();
            } catch (IOException e) {
                System.err.println("Error reading the CSV file '" + fileName + "': " + e.getMessage());
                continue; // Proceed to the next file
            }

            int n = instance.getNodes().size();
            if (n == 0) {
                System.out.println("No valid nodes found in the CSV file '" + fileName + "'. Skipping.");
                continue;
            }
            int k = (int) Math.ceil(n / 2.0);
            System.out.println("Total nodes: " + n + ", Selecting k=" + k + " nodes.\n");

            // Prepare output directory for this instance
            String outputInstanceDirPath = "outputs/" + instanceName;
            File outputInstanceDir = new File(outputInstanceDirPath);
            if (!outputInstanceDir.exists()) {
                boolean created = outputInstanceDir.mkdirs();
                if (!created) {
                    System.err.println("Failed to create output directory for instance '" + instanceName + "'. Skipping.");
                    continue;
                }
            }

            // Initialize a map to store solutions per method
            Map<String, List<Solution>> methodSolutions = new LinkedHashMap<>();
            for (Heuristic heuristic : heuristics) {
                String methodName = heuristic.getClass().getSimpleName();
                switch (methodName) {
                    case "RandomSolution":
                        methodName = "Random_Solution";
                        break;
                    case "NearestNeighborEnd":
                        methodName = "Nearest_Neighbor_End_Insertion";
                        break;
                    case "NearestNeighborAny":
                        methodName = "Nearest_Neighbor_Any_Insertion";
                        break;
                    case "GreedyCycle":
                        methodName = "Greedy_Cycle";
                        break;
                    default:
                        methodName = heuristic.getClass().getSimpleName();
                }
                methodSolutions.put(methodName, new ArrayList<>());
            }

            // Generate solutions for each heuristic and each start node
            for (Heuristic heuristic : heuristics) {
                String methodName = heuristic.getClass().getSimpleName();
                switch (methodName) {
                    case "RandomSolution":
                        methodName = "Random_Solution";
                        break;
                    case "NearestNeighborEnd":
                        methodName = "Nearest_Neighbor_End_Insertion";
                        break;
                    case "NearestNeighborAny":
                        methodName = "Nearest_Neighbor_Any_Insertion";
                        break;
                    case "GreedyCycle":
                        methodName = "Greedy_Cycle";
                        break;
                    default:
                        methodName = heuristic.getClass().getSimpleName();
                }
                System.out.println("Generating solutions using " + methodName + "...");
                List<Solution> solutions = new ArrayList<>();
                for (int startNode = 0; startNode < n; startNode++) {
                    Solution sol = heuristic.generateSolution(instance, k, startNode);
                    solutions.add(sol);
                }
                methodSolutions.get(methodName).addAll(solutions);
            }

            // Compute statistics for each method and save the best path
            System.out.println("\n--- Computational Experiment Results for Instance: " + instanceName + " ---\n");
            for (Map.Entry<String, List<Solution>> entry : methodSolutions.entrySet()) {
                String methodName = entry.getKey();
                List<Solution> solutions = entry.getValue();
                Statistics stats = Statistics.computeStatistics(solutions);
                System.out.println("Method: " + methodName);
                System.out.println("Min Objective: " + stats.getMinObjective());
                System.out.println("Max Objective: " + stats.getMaxObjective());
                System.out.printf("Average Objective: %.2f%n", stats.getAvgObjective());
                System.out.println("Best Solution Path: " + stats.getBestPath() + "\n");

                // Save best path to CSV
                String outputFileName = outputInstanceDirPath + "/" + methodName + ".csv";
                try {
                    saveBestPathToCSV(stats.getBestPath(), outputFileName);
                    System.out.println("Best path saved to " + outputFileName + "\n");
                } catch (IOException e) {
                    System.err.println("Error writing best path to CSV for method '" + methodName + "': " + e.getMessage());
                }
            }
            System.out.println("Finished processing instance: " + instanceName + "\n");
        }

        // After processing all instances, run the Python script
        String pythonScript = "plot_results.py";
        System.out.println("All instances processed. Executing '" + pythonScript + "'...");
        try {
            ProcessBuilder pb = new ProcessBuilder("python", pythonScript);
            pb.inheritIO(); // Redirect output and error streams to the console
            Process process = pb.start();
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("'" + pythonScript + "' executed successfully.");
            } else {
                System.err.println("'" + pythonScript + "' exited with code: " + exitCode);
            }
        } catch (Exception e) {
            System.err.println("Error executing '" + pythonScript + "': " + e.getMessage());
        }
    }

    /**
     * Saves the best path to a CSV file, with each node index on a separate line.
     * The first node is appended at the end to complete the cycle.
     * @param bestPath List of node indices representing the best path
     * @param fileName The name of the output CSV file
     * @throws IOException If file writing fails
     */
    private static void saveBestPathToCSV(List<Integer> bestPath, String fileName) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        for (Integer node : bestPath) {
            writer.write(node.toString());
            writer.newLine();
        }
        if (!bestPath.isEmpty()) {
            writer.write(bestPath.getFirst().toString());
        }
        writer.close();
    }
}