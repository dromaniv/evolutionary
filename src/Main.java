import java.io.*;
import java.util.*;

/**
 * Represents a node with x and y coordinates and a cost.
 */
class Node {
    final int x;
    final int y;
    final int cost;

    Node(int x, int y, int cost) {
        this.x = x;
        this.y = y;
        this.cost = cost;
    }
}

/**
 * Represents a problem instance with nodes and a distance matrix.
 */
class ProblemInstance {
    final Node[] nodes;
    int[][] distanceMatrix;

    ProblemInstance(List<Node> nodeList) {
        nodes = nodeList.toArray(new Node[0]);
    }

    /**
     * Reads nodes from a CSV file. Each line should have x;y;cost
     *
     * @param filePath Path to the CSV file
     * @return A ProblemInstance object
     * @throws IOException If file reading fails
     */
    static ProblemInstance readCSV(String filePath) throws IOException {
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
    void computeDistanceMatrix() {
        int n = nodes.length;
        distanceMatrix = new int[n][n];
        Node nodeI, nodeJ;
        for (int i = 0; i < n; i++) {
            nodeI = nodes[i];
            for (int j = i + 1; j < n; j++) {
                nodeJ = nodes[j];
                double dx = nodeI.x - nodeJ.x;
                double dy = nodeI.y - nodeJ.y;
                int roundedDist = (int) Math.round(Math.sqrt(dx * dx + dy * dy));
                distanceMatrix[i][j] = roundedDist;
                distanceMatrix[j][i] = roundedDist;
            }
        }
    }
}

/**
 * Represents a solution with a path and its objective value.
 */
class Solution {
    final int[] path;
    final int objectiveValue;

    Solution(int[] path, int objectiveValue) {
        this.path = Arrays.copyOf(path, path.length);
        this.objectiveValue = objectiveValue;
    }
}

/**
 * Abstract class for heuristic methods.
 */
abstract class Heuristic {
    final Random random = new Random();

    /**
     * Generates a solution based on the heuristic.
     *
     * @param instance  The problem instance
     * @param k         Number of nodes to select
     * @param startNode The starting node index
     * @return A Solution object
     */
    public abstract Solution generateSolution(ProblemInstance instance, int k, int startNode);

    /**
     * Computes the objective function: sum of path lengths + sum of node costs
     */
    int computeObjective(int[] path, ProblemInstance instance) {
        int totalDistance = 0;
        int k = path.length;
        int[][] distanceMatrix = instance.distanceMatrix;
        Node[] nodes = instance.nodes;
        int from, to;
        for (int i = 0; i < k; i++) {
            from = path[i];
            to = path[(i + 1) % k];
            totalDistance += distanceMatrix[from][to];
        }
        int totalCost = 0;
        for (int node : path) {
            totalCost += nodes[node].cost;
        }
        return totalDistance + totalCost;
    }
}

/**
 * Implements the Steepest Local Search with Move Evaluation and Edge Validity Checks.
 */
class SteepestLocalSearchWithMoveEvaluation extends Heuristic {

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        // For compatibility, we can set a default maxIterations if not specified
        return generateSolution(instance, k, startNode, Integer.MAX_VALUE);
    }

    /**
     * Generates a solution using local search with a maximum number of iterations.
     *
     * @param instance      The problem instance
     * @param k             Number of nodes to select
     * @param startNode     The starting node index
     * @param maxIterations Maximum number of iterations to perform
     * @return A Solution object
     */
    public Solution generateSolution(ProblemInstance instance, int k, int startNode, int maxIterations) {
        // Generate random initial path
        Node[] nodes = instance.nodes;
        int n = nodes.length;
        int[] allNodes = new int[n];
        for (int i = 0; i < n; i++) {
            allNodes[i] = i;
        }

        // Start with a random solution
        shuffleArray(allNodes);
        int[] currentPath = Arrays.copyOfRange(allNodes, 0, k);
        shuffleArray(currentPath);

        return generateSolutionFromPath(instance, k, startNode, currentPath, maxIterations);
    }

    /**
     * Generates a solution starting from an initial path, with a maximum number of iterations.
     *
     * @param instance      The problem instance
     * @param k             Number of nodes to select
     * @param startNode     The starting node index
     * @param initialPath   The initial path to start from
     * @param maxIterations Maximum number of iterations to perform
     * @return A Solution object
     */
    public Solution generateSolutionFromPath(ProblemInstance instance, int k, int startNode, int[] initialPath, int maxIterations) {
        int[] currentPath = Arrays.copyOf(initialPath, initialPath.length);

        int iterationCount = 0;
        boolean improvement = true;
        List<Move> improvingMoves = new ArrayList<>();
        int n = instance.nodes.length;
        boolean[] inPathSet = new boolean[n];
        for (int node : currentPath) {
            inPathSet[node] = true;
        }

        int[][] distanceMatrix = instance.distanceMatrix;
        Node[] nodes = instance.nodes;

        while (improvement && iterationCount < maxIterations) {
            iterationCount++;
            improvement = false;

            // Try improving moves from previous iterations
            Iterator<Move> iterator = improvingMoves.iterator();
            while (iterator.hasNext()) {
                Move move = iterator.next();
                if (!move.isValid(currentPath, inPathSet)) {
                    // Situation 1: Edges no longer exist; remove move from improvingMoves
                    iterator.remove();
                    continue;
                }

                int delta = move.computeDelta(currentPath, distanceMatrix, nodes);
                if (delta < 0) {
                    // Situation 3: Edges exist in the same orientation; apply move and remove from improvingMoves
                    move.apply(currentPath, inPathSet);
                    improvement = true;
                    iterator.remove();
                    break; // Apply one move at a time
                } else {
                    // Situation 2: Edges exist but do not lead to improvement; keep move in improvingMoves
                    continue;
                }
            }

            if (improvement) {
                continue;
            }

            // If no improvement from previous moves, explore full neighborhood
            int bestDelta = 0;
            Move bestMove = null;

            int nPath = currentPath.length;
            // Intra-route moves (2-opt)
            for (int i = 0; i < nPath; i++) {
                int a = currentPath[i];
                int b = currentPath[(i + 1) % nPath];
                for (int j = i + 2; j < nPath; j++) {
                    if (i == 0 && j == nPath - 1) {
                        continue; // Do not reverse the entire tour
                    }

                    int c = currentPath[j];
                    int d = currentPath[(j + 1) % nPath];

                    int delta = -distanceMatrix[a][b] - distanceMatrix[c][d]
                            + distanceMatrix[a][c] + distanceMatrix[b][d];

                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestMove = new Move();
                        bestMove.setIntraRouteMove(i, j, a, b, c, d);
                    }
                }
            }

            // Inter-route moves (Swap selected with unselected node)
            for (int i = 0; i < nPath; i++) {
                int u = currentPath[i];
                int prev = currentPath[(i - 1 + nPath) % nPath];
                int next = currentPath[(i + 1) % nPath];

                int distPrevU = distanceMatrix[prev][u];
                int distUNext = distanceMatrix[u][next];

                for (int v = 0; v < n; v++) {
                    if (inPathSet[v]) {
                        continue; // v is already in the path
                    }

                    int distPrevV = distanceMatrix[prev][v];
                    int distVNext = distanceMatrix[v][next];

                    int delta = -distPrevU - distUNext + distPrevV + distVNext
                            - nodes[u].cost + nodes[v].cost;

                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestMove = new Move();
                        bestMove.setInterRouteMove(i, v, u);
                    }
                }
            }

            if (bestMove != null) {
                // Apply best move
                bestMove.apply(currentPath, inPathSet);
                improvement = true;
                improvingMoves.clear();
                improvingMoves.add(new Move(bestMove)); // Save the move for future iterations
            }
        }

        int objective = computeObjective(currentPath, instance);
        return new Solution(currentPath, objective);
    }

    private void shuffleArray(int[] array) {
        int index, temp;
        for (int i = array.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    private static class Move {
        int index1;
        int index2;
        MoveType type;

        // For intra-route moves
        int nodeA, nodeB, nodeC, nodeD;

        // For inter-route moves
        int nodeU; // Original node at index1 when the move was created

        Move() {
        }

        Move(Move other) {
            this.index1 = other.index1;
            this.index2 = other.index2;
            this.type = other.type;
            this.nodeA = other.nodeA;
            this.nodeB = other.nodeB;
            this.nodeC = other.nodeC;
            this.nodeD = other.nodeD;
            this.nodeU = other.nodeU;
        }

        void setIntraRouteMove(int index1, int index2, int nodeA, int nodeB, int nodeC, int nodeD) {
            this.index1 = index1;
            this.index2 = index2;
            this.type = MoveType.INTRA_ROUTE;
            this.nodeA = nodeA;
            this.nodeB = nodeB;
            this.nodeC = nodeC;
            this.nodeD = nodeD;
        }

        void setInterRouteMove(int index1, int index2, int nodeU) {
            this.index1 = index1;
            this.index2 = index2;
            this.type = MoveType.INTER_ROUTE;
            this.nodeU = nodeU;
        }

        boolean isValid(int[] path, boolean[] inPathSet) {
            if (type == MoveType.INTRA_ROUTE) {
                int nPath = path.length;
                // Check if the nodes involved are still the same
                return path[index1] == nodeA && path[(index1 + 1) % nPath] == nodeB &&
                        path[index2] == nodeC && path[(index2 + 1) % nPath] == nodeD;
            } else if (type == MoveType.INTER_ROUTE) {
                // Check if the node at index1 is still nodeU and index2 is not in the path
                return path[index1] == nodeU && !inPathSet[index2];
            }
            return false;
        }

        int computeDelta(int[] path, int[][] distanceMatrix, Node[] nodes) {
            if (type == MoveType.INTRA_ROUTE) {
                int n = path.length;
                int a = path[index1];
                int b = path[(index1 + 1) % n];
                int c = path[index2];
                int d = path[(index2 + 1) % n];

                int delta = -distanceMatrix[a][b] - distanceMatrix[c][d]
                        + distanceMatrix[a][c] + distanceMatrix[b][d];
                return delta;
            } else if (type == MoveType.INTER_ROUTE) {
                int n = path.length;
                int u = path[index1];
                int v = index2;
                int prev = path[(index1 - 1 + n) % n];
                int next = path[(index1 + 1) % n];

                int delta = -distanceMatrix[prev][u] - distanceMatrix[u][next]
                        + distanceMatrix[prev][v] + distanceMatrix[v][next]
                        - nodes[u].cost + nodes[v].cost;
                return delta;
            }
            return Integer.MAX_VALUE;
        }

        void apply(int[] path, boolean[] inPathSet) {
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
            int temp;
            while (start < end) {
                temp = array[start];
                array[start] = array[end];
                array[end] = temp;
                start++;
                end--;
            }
        }
    }

    private enum MoveType {
        INTRA_ROUTE,
        INTER_ROUTE
    }
}

/**
 * Implements Multiple Start Local Search (MSLS).
 */
class MultipleStartLocalSearch extends Heuristic {

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        SteepestLocalSearchWithMoveEvaluation ls = new SteepestLocalSearchWithMoveEvaluation();

        // Perform local search starting from a random solution, limited to 200 iterations
        return ls.generateSolution(instance, k, startNode, 200);
    }
}

/**
 * Implements Iterated Local Search (ILS).
 */
class IteratedLocalSearch extends Heuristic {
    private final double maxTime; // in milliseconds
    public int numLocalSearches;

    public IteratedLocalSearch(double maxTime) {
        this.maxTime = maxTime;
    }

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        numLocalSearches = 0;
        long startTime = System.nanoTime();
        long maxDuration = (long) (maxTime * 1e6); // Convert milliseconds to nanoseconds

        // Ensure ILS runs for at least a minimum time
        long minDuration = 100 * 1_000_000; // 100 ms

        if (maxDuration < minDuration) {
            maxDuration = minDuration;
        }

        SteepestLocalSearchWithMoveEvaluation ls = new SteepestLocalSearchWithMoveEvaluation();

        // Start from a random solution
        Solution currentSolution = ls.generateSolution(instance, k, startNode, Integer.MAX_VALUE);
        numLocalSearches++;

        Solution bestSolution = currentSolution;
        Random random = new Random();

        while ((System.nanoTime() - startTime) < maxDuration) {
            // Perturb current solution
            int[] perturbedPath = perturbSolution(currentSolution.path, random);

            // Apply local search to the perturbed solution
            Solution newSolution = ls.generateSolutionFromPath(instance, k, startNode, perturbedPath, Integer.MAX_VALUE);
            numLocalSearches++;

            // Acceptance Criteria: Accept if better than current or with a probability
            if (newSolution.objectiveValue < bestSolution.objectiveValue) {
                bestSolution = newSolution;
                currentSolution = newSolution;
            } else {
                // Accept with a probability (e.g., 0.1) to allow exploration
                if (random.nextDouble() < 0.1) {
                    currentSolution = newSolution;
                }
            }
        }

        return bestSolution;
    }

    /**
     * Perturbation operator: Performs a double-bridge move.
     * This operator removes four edges and reconnects the segments differently.
     * It is effective in escaping local optima in TSP problems.
     *
     * @param path   The current solution path
     * @param random Random number generator
     * @return A new perturbed path
     */
    private int[] perturbSolution(int[] path, Random random) {
        int n = path.length;

        if (n < 6) {
            // For small n, perform a simple 2-opt perturbation
            return twoOptPerturbation(path, random);
        }

        // Generate four random, distinct integers in range [1, n - 2]
        TreeSet<Integer> indicesSet = new TreeSet<>();
        while (indicesSet.size() < 4) {
            indicesSet.add(1 + random.nextInt(n - 3)); // random integer from 1 to n - 3 inclusive
        }
        List<Integer> indicesList = new ArrayList<>(indicesSet);
        int i = indicesList.get(0);
        int j = indicesList.get(1);
        int k = indicesList.get(2);
        int l = indicesList.get(3);

        // Create new path by swapping segments
        int[] newPath = new int[n];
        int pos = 0;

        // Copy Segment1: path[0 to i - 1]
        for (int idx = 0; idx < i; idx++) {
            newPath[pos++] = path[idx];
        }

        // Copy Segment3: path[j to k - 1]
        for (int idx = j; idx < k; idx++) {
            newPath[pos++] = path[idx];
        }

        // Copy Segment2: path[i to j - 1]
        for (int idx = i; idx < j; idx++) {
            newPath[pos++] = path[idx];
        }

        // Copy Segment4: path[k to l - 1]
        for (int idx = k; idx < l; idx++) {
            newPath[pos++] = path[idx];
        }

        // Copy Segment5: path[l to n - 1]
        for (int idx = l; idx < n; idx++) {
            newPath[pos++] = path[idx];
        }

        return newPath;
    }

    /**
     * Performs a simple 2-opt perturbation by reversing a subsequence.
     * This is used when the instance size is too small for a double-bridge move.
     *
     * @param path   The current solution path
     * @param random Random number generator
     * @return A new perturbed path
     */
    private int[] twoOptPerturbation(int[] path, Random random) {
        int n = path.length;
        int i = random.nextInt(n);
        int j = random.nextInt(n);
        while (j == i) {
            j = random.nextInt(n);
        }

        if (i > j) {
            int temp = i;
            i = j;
            j = temp;
        }

        int[] newPath = Arrays.copyOf(path, n);

        // Reverse the subsequence between i and j
        while (i < j) {
            int temp = newPath[i];
            newPath[i] = newPath[j];
            newPath[j] = temp;
            i++;
            j--;
        }

        return newPath;
    }
}

/**
 * Utility class to compute statistics for a list of solutions.
 */
class Statistics {
    final int minObjective;
    final int maxObjective;
    final double avgObjective;
    final int[] bestPath;

    Statistics(int minObjective, int maxObjective, double avgObjective, int[] bestPath) {
        this.minObjective = minObjective;
        this.maxObjective = maxObjective;
        this.avgObjective = avgObjective;
        this.bestPath = bestPath;
    }

    /**
     * Computes statistics from a list of solutions.
     *
     * @param solutions List of Solution objects
     * @return A Statistics object
     */
    static Statistics computeStatistics(List<Solution> solutions) {
        if (solutions == null || solutions.isEmpty()) {
            return new Statistics(0, 0, 0.0, new int[0]);
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        long sum = 0;
        int[] bestPath = new int[0];
        for (Solution sol : solutions) {
            int obj = sol.objectiveValue;
            if (obj < min) {
                min = obj;
                bestPath = sol.path;
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

            int n = instance.nodes.length;
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

            // Run Multiple Start Local Search (MSLS)
            System.out.println("Running Multiple Start Local Search (MSLS)...");
            List<Solution> mslsSolutions = new ArrayList<>();
            double totalMslsTime = 0.0;
            for (int run = 0; run < 20; run++) {
                MultipleStartLocalSearch msls = new MultipleStartLocalSearch();
                long startTime = System.nanoTime();
                Solution sol = msls.generateSolution(instance, k, 0);
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;
                totalMslsTime += durationMs;
                mslsSolutions.add(sol);
            }
            double mslsAvgTime = totalMslsTime / 20;

            // Ensure minimum runtime for ILS
            double ilsRunTime = Math.max(mslsAvgTime, 100.0); // At least 100 ms

            // Run Iterated Local Search (ILS)
            System.out.println("Running Iterated Local Search (ILS)...");
            List<Solution> ilsSolutions = new ArrayList<>();
            double totalIlsTime = 0.0;
            List<Integer> ilsNumLocalSearchesList = new ArrayList<>();
            for (int run = 0; run < 20; run++) {
                IteratedLocalSearch ils = new IteratedLocalSearch(ilsRunTime);
                long startTime = System.nanoTime();
                Solution sol = ils.generateSolution(instance, k, 0);
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;
                totalIlsTime += durationMs;
                ilsSolutions.add(sol);
                ilsNumLocalSearchesList.add(ils.numLocalSearches);
            }
            double avgIlsTime = totalIlsTime / 20;
            double totalNumLocalSearches = 0;
            for (int numLS : ilsNumLocalSearchesList) {
                totalNumLocalSearches += numLS;
            }
            double avgNumLocalSearches = totalNumLocalSearches / 20;

            // Compute statistics for MSLS
            Statistics mslsStats = Statistics.computeStatistics(mslsSolutions);
            // Compute statistics for ILS
            Statistics ilsStats = Statistics.computeStatistics(ilsSolutions);

            // Output results
            System.out.println("\n--- Computational Experiment Results for Instance: " + instanceName + " ---\n");

            // Results for MSLS
            System.out.println("Method: Multiple Start Local Search (MSLS)");
            System.out.println("Min Objective: " + mslsStats.minObjective);
            System.out.println("Max Objective: " + mslsStats.maxObjective);
            System.out.printf("Average Objective: %.2f%n", mslsStats.avgObjective);
            System.out.printf("Average Time per run: %.2f ms%n", mslsAvgTime);
            System.out.println("Best Solution Path: " + Arrays.toString(mslsStats.bestPath) + "\n");

            // Save best path for MSLS
            String mslsOutputFileName = outputInstanceDirPath + "/MSLS.csv";
            try {
                saveBestPathToCSV(mslsStats.bestPath, mslsOutputFileName);
                System.out.println("Best path for MSLS saved to " + mslsOutputFileName + "\n");
            } catch (IOException e) {
                System.err.println("Error writing best path to CSV for MSLS: " + e.getMessage());
            }

            // Results for ILS
            System.out.println("Method: Iterated Local Search (ILS)");
            System.out.println("Min Objective: " + ilsStats.minObjective);
            System.out.println("Max Objective: " + ilsStats.maxObjective);
            System.out.printf("Average Objective: %.2f%n", ilsStats.avgObjective);
            System.out.printf("Average Time per run: %.2f ms%n", avgIlsTime);
            System.out.printf("Average number of local searches: %.2f%n", avgNumLocalSearches);
            System.out.println("Best Solution Path: " + Arrays.toString(ilsStats.bestPath) + "\n");

            // Save best path for ILS
            String ilsOutputFileName = outputInstanceDirPath + "/ILS.csv";
            try {
                saveBestPathToCSV(ilsStats.bestPath, ilsOutputFileName);
                System.out.println("Best path for ILS saved to " + ilsOutputFileName + "\n");
            } catch (IOException e) {
                System.err.println("Error writing best path to CSV for ILS: " + e.getMessage());
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
     *
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