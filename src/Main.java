import java.io.*;
import java.util.*;

/**
 * Represents a node with x and y coordinates and a cost.
 */
class Node {
    private final int x;
    private final int y;
    private final int cost;

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
    private final List<Node> nodes;
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
    private final List<Integer> path;
    private final int objectiveValue;

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

    /**
     * Computes the objective function: sum of path lengths + sum of node costs
     */
    protected int computeObjective(List<Integer> path, ProblemInstance instance) {
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

    protected static class InsertionPosition {
        int position;
        int increase;

        public InsertionPosition(int position, int increase) {
            this.position = position;
            this.increase = increase;
        }
    }

    protected static class InsertionInfo {
        int position;
        int bestIncrease;
        int secondBestIncrease;

        public InsertionInfo(int position, int bestIncrease, int secondBestIncrease) {
            this.position = position;
            this.bestIncrease = bestIncrease;
            this.secondBestIncrease = secondBestIncrease;
        }
    }

    protected InsertionPosition findBestInsertionPosition(List<Integer> path, int nodeToInsert, int[][] distanceMatrix) {
        int bestPos = -1;
        int minIncrease = Integer.MAX_VALUE;
        int pathSize = path.size();
        for (int i = 0; i < pathSize; i++) {
            int current = path.get(i);
            int next = path.get((i + 1) % pathSize);
            int increase = distanceMatrix[current][nodeToInsert] + distanceMatrix[nodeToInsert][next] - distanceMatrix[current][next];
            if (increase < minIncrease) {
                minIncrease = increase;
                bestPos = i + 1;
            }
        }
        return new InsertionPosition(bestPos, minIncrease);
    }

    protected InsertionInfo findBestAndSecondBestInsertion(List<Integer> path, int nodeToInsert, int[][] distanceMatrix) {
        int bestIncrease = Integer.MAX_VALUE;
        int secondBestIncrease = Integer.MAX_VALUE;
        int bestPos = -1;

        for (int i = 0; i < path.size(); i++) {
            int current = path.get(i);
            int next = path.get((i + 1) % path.size());
            int increase = distanceMatrix[current][nodeToInsert] + distanceMatrix[nodeToInsert][next] - distanceMatrix[current][next];

            if (increase < bestIncrease) {
                secondBestIncrease = bestIncrease;
                bestIncrease = increase;
                bestPos = i + 1;
            } else if (increase < secondBestIncrease) {
                secondBestIncrease = increase;
            }
        }
        return new InsertionInfo(bestPos, bestIncrease, secondBestIncrease);
    }
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
        selectedNodes = selectedNodes.subList(0, Math.min(k, selectedNodes.size()));
        int objective = computeObjective(selectedNodes, instance);
        return new Solution(selectedNodes, objective);
    }
}

/**
 * Implements the Nearest Neighbor heuristic by adding nodes to the end of the path,
 * considering the sum of Euclidean distance and node cost.
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
            int last = path.getLast();
            int nearest = -1;
            int minDist = Integer.MAX_VALUE;

            for (int node = 0; node < n; node++) {
                if (!selected.contains(node)) {
                    int dist = distanceMatrix[last][node];
                    int effectiveDistance = dist + nodes.get(node).getCost();
                    if (effectiveDistance < minDist) {
                        minDist = effectiveDistance;
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
}

/**
 * Implements the Nearest Neighbor heuristic by adding nodes at any position in the path,
 * considering the sum of Euclidean distance and node cost.
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
                        int effectiveDistance = dist + nodes.get(node).getCost();
                        if (effectiveDistance < minDist) {
                            minDist = effectiveDistance;
                            nearest = node;
                        }
                    }
                }
            }

            if (nearest != -1) {
                InsertionPosition insertion = findBestInsertionPosition(path, nearest, distanceMatrix);
                path.add(insertion.position, nearest);
                selected.add(nearest);
            } else {
                break; // No more nodes to add
            }
        }

        int objective = computeObjective(path, instance);
        return new Solution(path, objective);
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
                    InsertionPosition insertion = findBestInsertionPosition(path, node, distanceMatrix);
                    int minIncrease = insertion.increase;
                    int score = minIncrease + nodes.get(node).getCost();
                    if (score < bestScore) {
                        bestScore = score;
                        bestNode = node;
                    }
                }
            }
            if (bestNode != -1) {
                InsertionPosition insertion = findBestInsertionPosition(path, bestNode, distanceMatrix);
                if (insertion.position != -1) {
                    path.add(insertion.position, bestNode);
                    selected.add(bestNode);
                } else {
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
}

/**
 * Implements the Greedy 2-Regret heuristic.
 */
class Greedy2Regret extends Heuristic {

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
            int maxRegretNode = -1;
            int maxRegretValue = Integer.MIN_VALUE;
            int bestInsertionPosition = -1;

            for (int node = 0; node < n; node++) {
                if (selected.contains(node)) {
                    continue;
                }
                int bestIncrease = Integer.MAX_VALUE;
                int secondBestIncrease = Integer.MAX_VALUE;
                int bestPos = -1;

                // Evaluate all possible insertion positions
                for (int i = 0; i < path.size(); i++) {
                    int current = path.get(i);
                    int next = path.get((i + 1) % path.size());
                    int increase = distanceMatrix[current][node] + distanceMatrix[node][next] - distanceMatrix[current][next] + nodes.get(node).getCost();

                    if (increase < bestIncrease) {
                        secondBestIncrease = bestIncrease;
                        bestIncrease = increase;
                        bestPos = i + 1;
                    } else if (increase < secondBestIncrease) {
                        secondBestIncrease = increase;
                    }
                }

                int regretValue = secondBestIncrease - bestIncrease;
                if (regretValue > maxRegretValue) {
                    maxRegretValue = regretValue;
                    maxRegretNode = node;
                    bestInsertionPosition = bestPos;
                }
            }

            if (maxRegretNode != -1 && bestInsertionPosition != -1) {
                path.add(bestInsertionPosition, maxRegretNode);
                selected.add(maxRegretNode);
            } else {
                break; // No more nodes to add
            }
        }

        int objective = computeObjective(path, instance);
        return new Solution(path, objective);
    }
}

/**
 * Implements the Greedy heuristic with weighted sum criterion (2-Regret + Best Increase).
 */
class GreedyWeightedRegret extends Heuristic {
    private final double w1; // Weight for regret
    private final double w2; // Weight for best increase

    public GreedyWeightedRegret() {
        this.w1 = 1;
        this.w2 = 1;
    }

    public GreedyWeightedRegret(double w1, double w2) {
        this.w1 = w1;
        this.w2 = w2;
    }

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
            double bestWeightedValue = Double.NEGATIVE_INFINITY;
            int bestInsertionPosition = -1;

            for (int node = 0; node < n; node++) {
                if (selected.contains(node)) {
                    continue;
                }
                InsertionInfo insertionInfo = findBestAndSecondBestInsertion(path, node, distanceMatrix);

                int regretValue = insertionInfo.secondBestIncrease - insertionInfo.bestIncrease;
                double weightedValue = w1 * regretValue - w2 * (insertionInfo.bestIncrease + nodes.get(node).getCost());

                if (weightedValue > bestWeightedValue) {
                    bestWeightedValue = weightedValue;
                    bestNode = node;
                    bestInsertionPosition = insertionInfo.position;
                }
            }

            if (bestNode != -1 && bestInsertionPosition != -1) {
                path.add(bestInsertionPosition, bestNode);
                selected.add(bestNode);
            } else {
                break; // No more nodes to add
            }
        }

        int objective = computeObjective(path, instance);
        return new Solution(path, objective);
    }
}

/**
 * Utility class to compute statistics for a list of solutions.
 */
class Statistics {
    private final int minObjective;
    private final int maxObjective;
    private final double avgObjective;
    private final List<Integer> bestPath;

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
        File[] inputFiles = inputDir.listFiles((__, name) -> name.toLowerCase().endsWith(".csv"));

        // Sort the files by name
        if (inputFiles != null) {
            Arrays.sort(inputFiles, Comparator.comparing(File::getName));
        }

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
        heuristics.add(new Greedy2Regret());
        heuristics.add(new GreedyWeightedRegret());

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
            Map<String, Double> methodTimes = new LinkedHashMap<>();
            for (Heuristic heuristic : heuristics) {
                String methodName = heuristic.getClass().getSimpleName();
                methodSolutions.put(methodName, new ArrayList<>());
            }

            // Generate solutions for each heuristic and each start node
            for (Heuristic heuristic : heuristics) {
                String methodName = heuristic.getClass().getSimpleName();
                System.out.println("Generating solutions using " + methodName + "...");
                List<Solution> solutions = new ArrayList<>();

                // Start timing
                long startTime = System.nanoTime();

                for (int startNode = 0; startNode < n; startNode++) {
                    Solution sol = heuristic.generateSolution(instance, k, startNode);
                    solutions.add(sol);
                }

                // End timing
                long endTime = System.nanoTime();
                long duration = endTime - startTime;
                double durationMs = duration / 1e6;

                methodSolutions.get(methodName).addAll(solutions);
                methodTimes.put(methodName, durationMs);
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
                System.out.printf("Time taken: %.2f ms%n", methodTimes.get(methodName));
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
        String pythonScript = "Evolutionary Computation\\plots_results.py";
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