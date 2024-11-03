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
    private Map<Integer, List<Integer>> candidateNeighbors;

    public ProblemInstance() {
        nodes = new ArrayList<>();
        candidateNeighbors = new HashMap<>();
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

    /**
     * Computes candidate neighbors for each node based on the sum of edge length and node cost.
     * @param numNeighbors Number of candidate neighbors to consider
     */
    public void computeCandidateNeighbors(int numNeighbors) {
        int n = nodes.size();
        candidateNeighbors = new HashMap<>();
        for (int i = 0; i < n; i++) {
            List<Integer> neighbors = new ArrayList<>();
            List<Pair> neighborPairs = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                Node nodeJ = nodes.get(j);
                int distance = distanceMatrix[i][j];
                int effectiveDistance = distance + nodeJ.getCost();
                neighborPairs.add(new Pair(j, effectiveDistance));
            }
            neighborPairs.sort(Comparator.comparingInt(p -> p.distance));
            for (int k = 0; k < Math.min(numNeighbors, neighborPairs.size()); k++) {
                neighbors.add(neighborPairs.get(k).nodeIndex);
            }
            candidateNeighbors.put(i, neighbors);
        }
    }

    private static class Pair {
        int nodeIndex;
        int distance;

        public Pair(int nodeIndex, int distance) {
            this.nodeIndex = nodeIndex;
            this.distance = distance;
        }
    }

    public List<Node> getNodes() {
        return nodes;
    }

    public int[][] getDistanceMatrix() {
        return distanceMatrix;
    }

    public Map<Integer, List<Integer>> getCandidateNeighbors() {
        return candidateNeighbors;
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
    protected Random random = new Random(42);

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
}

/**
 * Implements the Steepest Local Search with Candidate Moves heuristic.
 */
class SteepestLocalSearchWithCandidates extends Heuristic {
    private final int numCandidateNeighbors;

    public SteepestLocalSearchWithCandidates(int numCandidateNeighbors) {
        this.numCandidateNeighbors = numCandidateNeighbors;
    }

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        List<Node> nodes = instance.getNodes();
        int n = nodes.size();
        List<Integer> allNodes = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            allNodes.add(i);
        }

        // Start with a random solution
        Collections.shuffle(allNodes, random);
        List<Integer> currentPath = new ArrayList<>(allNodes.subList(0, k));
        // Create a random tour
        Collections.shuffle(currentPath, random);

        int[][] distanceMatrix = instance.getDistanceMatrix();
        Map<Integer, List<Integer>> candidateNeighbors = instance.getCandidateNeighbors();
        Set<Integer> inPathSet = new HashSet<>(currentPath);

        boolean improvement = true;
        while (improvement) {
            improvement = false;
            int bestDelta = 0;
            Move bestMove = null;

            // Intra-route moves (2-opt)
            for (int i = 0; i < currentPath.size(); i++) {
                int u = currentPath.get(i);
                List<Integer> uCandidates = candidateNeighbors.get(u);

                for (int v : uCandidates) {
                    int j = currentPath.indexOf(v);
                    if (j == -1 || Math.abs(i - j) == 1 || Math.abs(i - j) == currentPath.size() - 1) {
                        continue; // v is not in path or u and v are adjacent
                    }

                    // Compute delta objective for 2-opt move between u and v
                    int delta = compute2OptDelta(currentPath, i, j, distanceMatrix, nodes);

                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestMove = new Move(i, j, delta, MoveType.INTRA_ROUTE);
                        improvement = true;
                    }
                }
            }

            // Inter-route moves (Swap selected with unselected node)
            for (int i = 0; i < currentPath.size(); i++) {
                int u = currentPath.get(i);
                List<Integer> uCandidates = candidateNeighbors.get(u);

                for (int v : uCandidates) {
                    if (inPathSet.contains(v)) {
                        continue; // v is already in the path
                    }

                    // Compute delta objective for swapping u with v
                    int delta = computeSwapDelta(currentPath, i, v, distanceMatrix, nodes, candidateNeighbors);

                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestMove = new Move(i, v, delta, MoveType.INTER_ROUTE);
                        improvement = true;
                    }
                }
            }

            if (improvement && bestMove != null) {
                if (bestMove.type == MoveType.INTRA_ROUTE) {
                    // Apply 2-opt move
                    int i = bestMove.index1;
                    int j = bestMove.index2;
                    if (i > j) {
                        int temp = i;
                        i = j;
                        j = temp;
                    }
                    Collections.reverse(currentPath.subList(i + 1, j + 1));
                } else if (bestMove.type == MoveType.INTER_ROUTE) {
                    // Apply swap move
                    int i = bestMove.index1;
                    int u = currentPath.get(i);
                    int v = bestMove.index2;

                    currentPath.set(i, v);
                    inPathSet.remove(u);
                    inPathSet.add(v);
                }
            }
        }

        int objective = computeObjective(currentPath, instance);
        return new Solution(currentPath, objective);
    }

    private int compute2OptDelta(List<Integer> path, int i, int j, int[][] distanceMatrix, List<Node> nodes) {
        int n = path.size();
        int a = path.get(i);
        int b = path.get((i + 1) % n);
        int c = path.get(j);
        int d = path.get((j + 1) % n);

        int delta = -distanceMatrix[a][b] - distanceMatrix[c][d] + distanceMatrix[a][c] + distanceMatrix[b][d];
        // Node costs remain the same
        return delta;
    }

    private int computeSwapDelta(List<Integer> path, int i, int v, int[][] distanceMatrix, List<Node> nodes, Map<Integer, List<Integer>> candidateNeighbors) {
        int n = path.size();
        int u = path.get(i);
        int prev = path.get((i - 1 + n) % n);
        int next = path.get((i + 1) % n);

        int delta = -distanceMatrix[prev][u] - distanceMatrix[u][next] + distanceMatrix[prev][v] + distanceMatrix[v][next];
        delta -= nodes.get(u).getCost();
        delta += nodes.get(v).getCost();

        // Check if at least one of the new edges is a candidate edge
        boolean candidateEdgeIntroduced = candidateNeighbors.get(prev).contains(v) || candidateNeighbors.get(v).contains(next);
        if (!candidateEdgeIntroduced) {
            return Integer.MAX_VALUE; // Invalid move
        }

        return delta;
    }

    private static class Move {
        int index1;
        int index2;
        int delta;
        MoveType type;

        public Move(int index1, int index2, int delta, MoveType type) {
            this.index1 = index1;
            this.index2 = index2;
            this.delta = delta;
            this.type = type;
        }
    }

    private enum MoveType {
        INTRA_ROUTE,
        INTER_ROUTE
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
        File[] inputFiles = inputDir.listFiles((_, name) -> name.toLowerCase().endsWith(".csv"));

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
        heuristics.add(new SteepestLocalSearchWithCandidates(10)); // 10 candidate neighbors

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

                System.out.println("Computing candidate neighbors...");
                instance.computeCandidateNeighbors(10);
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

            // Generate solutions for each heuristic and run 200 times
            for (Heuristic heuristic : heuristics) {
                String methodName = heuristic.getClass().getSimpleName();
                System.out.println("Generating solutions using " + methodName + "...");
                List<Solution> solutions = new ArrayList<>();

                // Start timing
                long startTime = System.nanoTime();

                for (int run = 0; run < 200; run++) {
                    Solution sol = heuristic.generateSolution(instance, k, 0);
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
            writer.write(bestPath.get(0).toString());
        }
        writer.close();
    }
}