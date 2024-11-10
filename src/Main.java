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
    private final Node[] nodes;
    private int[][] distanceMatrix;

    public ProblemInstance(List<Node> nodeList) {
        nodes = nodeList.toArray(new Node[0]);
    }

    /**
     * Reads nodes from a CSV file. Each line should have x;y;cost
     * @param filePath Path to the CSV file
     * @return A ProblemInstance object
     * @throws IOException If file reading fails
     */
    public static ProblemInstance readCSV(String filePath) throws IOException {
        List<Node> nodeList = new ArrayList<>();
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
                nodeList.add(new Node(x, y, cost));
            } catch (NumberFormatException e) {
                // Skip rows with invalid integers
            }
        }
        reader.close();
        return new ProblemInstance(nodeList);
    }

    /**
     * Computes the Euclidean distance matrix between nodes, rounded to the nearest integer.
     */
    public void computeDistanceMatrix() {
        int n = nodes.length;
        distanceMatrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            Node nodeI = nodes[i];
            for (int j = i + 1; j < n; j++) {
                Node nodeJ = nodes[j];
                double dist = Math.sqrt(Math.pow(nodeI.getX() - nodeJ.getX(), 2) +
                        Math.pow(nodeI.getY() - nodeJ.getY(), 2));
                int roundedDist = (int) Math.round(dist);
                distanceMatrix[i][j] = roundedDist;
                distanceMatrix[j][i] = roundedDist;
            }
        }
    }

    public Node[] getNodes() {
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
    private final int[] path;
    private final int objectiveValue;

    public Solution(int[] path, int objectiveValue) {
        this.path = Arrays.copyOf(path, path.length);
        this.objectiveValue = objectiveValue;
    }

    public int[] getPath() {
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
    protected int computeObjective(int[] path, ProblemInstance instance) {
        int totalDistance = 0;
        int k = path.length;
        int[][] distanceMatrix = instance.getDistanceMatrix();
        Node[] nodes = instance.getNodes();
        for (int i = 0; i < k; i++) {
            int from = path[i];
            int to = path[(i + 1) % k];
            totalDistance += distanceMatrix[from][to];
        }
        int totalCost = 0;
        for (int node : path) {
            totalCost += nodes[node].getCost();
        }
        return totalDistance + totalCost;
    }
}

/**
 * Implements the Steepest Local Search with Move Evaluation.
 */
class SteepestLocalSearchWithMoveEvaluation extends Heuristic {

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        Node[] nodes = instance.getNodes();
        int n = nodes.length;
        int[] allNodes = new int[n];
        for (int i = 0; i < n; i++) {
            allNodes[i] = i;
        }

        // Start with a random solution
        int[] currentPath = Arrays.copyOfRange(allNodes, 0, k);
        shuffleArray(currentPath);

        int[][] distanceMatrix = instance.getDistanceMatrix();
        boolean[] inPathSet = new boolean[n];
        for (int node : currentPath) {
            inPathSet[node] = true;
        }

        boolean improvement = true;
        List<Move> improvingMoves = new ArrayList<>();

        while (improvement) {
            improvement = false;
            List<Move> newImprovingMoves = new ArrayList<>();

            // Try improving moves from previous iterations
            for (Move move : improvingMoves) {
                if (!move.isValid(currentPath, inPathSet, n)) {
                    continue;
                }

                int delta = move.computeDelta(currentPath, distanceMatrix, nodes);
                if (delta < 0) {
                    // Apply move
                    move.apply(currentPath, inPathSet);
                    improvement = true;
                    newImprovingMoves.add(move);
                    break; // Apply one move at a time
                }
            }

            if (improvement) {
                improvingMoves = newImprovingMoves;
                continue;
            }

            // If no improvement from previous moves, explore full neighborhood
            int bestDelta = 0;
            Move bestMove = null;

            // Intra-route moves (2-opt)
            for (int i = 0; i < currentPath.length; i++) {
                for (int j = i + 2; j < currentPath.length; j++) {
                    if (i == 0 && j == currentPath.length - 1) {
                        continue; // Do not reverse the entire tour
                    }

                    int delta = compute2OptDelta(currentPath, i, j, distanceMatrix);

                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestMove = new Move(i, j, MoveType.INTRA_ROUTE);
                    }
                }
            }

            // Inter-route moves (Swap selected with unselected node)
            for (int i = 0; i < currentPath.length; i++) {

                for (int v = 0; v < n; v++) {
                    if (inPathSet[v]) {
                        continue; // v is already in the path
                    }

                    int delta = computeSwapDelta(currentPath, i, v, distanceMatrix, nodes);

                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestMove = new Move(i, v, MoveType.INTER_ROUTE);
                    }
                }
            }

            if (bestMove != null) {
                // Apply best move
                bestMove.apply(currentPath, inPathSet);
                improvement = true;
                improvingMoves.clear();
                improvingMoves.add(bestMove);
            }
        }

        int objective = computeObjective(currentPath, instance);
        return new Solution(currentPath, objective);
    }

    private void shuffleArray(int[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);
            // Simple swap
            int temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    private int compute2OptDelta(int[] path, int i, int j, int[][] distanceMatrix) {
        int n = path.length;
        int a = path[i];
        int b = path[(i + 1) % n];
        int c = path[j];
        int d = path[(j + 1) % n];

        int delta = -distanceMatrix[a][b] - distanceMatrix[c][d] + distanceMatrix[a][c] + distanceMatrix[b][d];
        return delta;
    }

    private int computeSwapDelta(int[] path, int i, int v, int[][] distanceMatrix, Node[] nodes) {
        int n = path.length;
        int u = path[i];
        int prev = path[(i - 1 + n) % n];
        int next = path[(i + 1) % n];

        int delta = -distanceMatrix[prev][u] - distanceMatrix[u][next] + distanceMatrix[prev][v] + distanceMatrix[v][next];
        delta -= nodes[u].getCost();
        delta += nodes[v].getCost();

        return delta;
    }

    private static class Move {
        int index1;
        int index2;
        MoveType type;

        public Move(int index1, int index2, MoveType type) {
            this.index1 = index1;
            this.index2 = index2;
            this.type = type;
        }

        public boolean isValid(int[] path, boolean[] inPathSet, int n) {
            if (type == MoveType.INTRA_ROUTE) {
                return index1 >= 0 && index1 < path.length && index2 >= 0 && index2 < path.length;
            } else if (type == MoveType.INTER_ROUTE) {
                return index1 >= 0 && index1 < path.length && !inPathSet[index2] && index2 >= 0 && index2 < n;
            }
            return false;
        }

        public int computeDelta(int[] path, int[][] distanceMatrix, Node[] nodes) {
            if (type == MoveType.INTRA_ROUTE) {
                return compute2OptDelta(path, index1, index2, distanceMatrix);
            } else if (type == MoveType.INTER_ROUTE) {
                return computeSwapDelta(path, index1, index2, distanceMatrix, nodes);
            }
            return Integer.MAX_VALUE;
        }

        public void apply(int[] path, boolean[] inPathSet) {
            if (type == MoveType.INTRA_ROUTE) {
                int i = index1;
                int j = index2;
                if (i > j) {
                    int temp = i;
                    i = j;
                    j = temp;
                }
                reverseSubarray(path, i + 1, j);
            } else if (type == MoveType.INTER_ROUTE) {
                int i = index1;
                int u = path[i];
                int v = index2;

                path[i] = v;
                inPathSet[u] = false;
                inPathSet[v] = true;
            }
        }

        private void reverseSubarray(int[] array, int start, int end) {
            while (start < end) {
                int temp = array[start];
                array[start] = array[end];
                array[end] = temp;
                start++;
                end--;
            }
        }

        private int compute2OptDelta(int[] path, int i, int j, int[][] distanceMatrix) {
            int n = path.length;
            int a = path[i];
            int b = path[(i + 1) % n];
            int c = path[j];
            int d = path[(j + 1) % n];

            int delta = -distanceMatrix[a][b] - distanceMatrix[c][d] + distanceMatrix[a][c] + distanceMatrix[b][d];
            return delta;
        }

        private int computeSwapDelta(int[] path, int i, int v, int[][] distanceMatrix, Node[] nodes) {
            int n = path.length;
            int u = path[i];
            int prev = path[(i - 1 + n) % n];
            int next = path[(i + 1) % n];

            int delta = -distanceMatrix[prev][u] - distanceMatrix[u][next] + distanceMatrix[prev][v] + distanceMatrix[v][next];
            delta -= nodes[u].getCost();
            delta += nodes[v].getCost();

            return delta;
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
    private final int[] bestPath;

    public Statistics(int minObjective, int maxObjective, double avgObjective, int[] bestPath) {
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

    public int[] getBestPath() {
        return bestPath;
    }

    /**
     * Computes statistics from a list of solutions.
     * @param solutions List of Solution objects
     * @return A Statistics object
     */
    public static Statistics computeStatistics(List<Solution> solutions) {
        if (solutions == null || solutions.isEmpty()) {
            return new Statistics(0, 0, 0.0, new int[0]);
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        long sum = 0;
        int[] bestPath = new int[0];
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
        heuristics.add(new SteepestLocalSearchWithMoveEvaluation());

        // Iterate over each input file
        for (File inputFile : inputFiles) {
            String fileName = inputFile.getName();
            String instanceName = fileName.substring(0, fileName.lastIndexOf('.')); // Remove .csv extension
            String filePath = inputFile.getPath();

            System.out.println("Processing instance: " + instanceName);

            // Initialize problem instance
            ProblemInstance instance;
            try {
                System.out.println("Reading nodes from CSV...");
                instance = ProblemInstance.readCSV(filePath);

                System.out.println("Computing distance matrix...");
                instance.computeDistanceMatrix();

            } catch (IOException e) {
                System.err.println("Error reading the CSV file '" + fileName + "': " + e.getMessage());
                continue; // Proceed to the next file
            }

            int n = instance.getNodes().length;
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
                System.out.println("Best Solution Path: " + Arrays.toString(stats.getBestPath()) + "\n");

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
     * @param bestPath Array of node indices representing the best path
     * @param fileName The name of the output CSV file
     * @throws IOException If file writing fails
     */
    private static void saveBestPathToCSV(int[] bestPath, String fileName) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        for (int node : bestPath) {
            writer.write(Integer.toString(node));
            writer.newLine();
        }
        if (bestPath.length > 0) {
            writer.write(Integer.toString(bestPath[0]));
        }
        writer.close();
    }
}