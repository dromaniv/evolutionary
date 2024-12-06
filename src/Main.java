import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

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

    /**
     * Generates a solution using local search with a maximum number of iterations.
     *
     * @param instance      The problem instance
     * @param initialPath   The initial path to start from
     * @param maxIterations Maximum number of iterations to perform
     * @return A Solution object
     */
    public Solution generateSolutionFromPath(ProblemInstance instance, int[] initialPath, int maxIterations) {
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

            // Inter-route moves (swap selected with unselected)
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
                improvingMoves.add(new Move(bestMove));
            }
        }

        int objective = computeObjective(currentPath, instance);
        return new Solution(currentPath, objective);
    }

    /**
     * Generates a random initial solution and then applies local search.
     *
     * @param instance      The problem instance
     * @param k             Number of nodes to select
     * @param maxIterations Maximum iterations for local search
     * @return A Solution object
     */
    public Solution generateSolution(ProblemInstance instance, int k, int maxIterations) {
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

        return generateSolutionFromPath(instance, currentPath, maxIterations);
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
        int nodeU; // Original node at index1

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

                return -distanceMatrix[a][b] - distanceMatrix[c][d]
                        + distanceMatrix[a][c] + distanceMatrix[b][d];

            } else if (type == MoveType.INTER_ROUTE) {
                int n = path.length;
                int u = path[index1];
                int v = index2;
                int prev = path[(index1 - 1 + n) % n];
                int next = path[(index1 + 1) % n];

                return -distanceMatrix[prev][u] - distanceMatrix[u][next]
                        + distanceMatrix[prev][v] + distanceMatrix[v][next]
                        - nodes[u].cost + nodes[v].cost;
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
 * Implements the Greedy Weighted Regret heuristic for solution construction (repair).
 * It uses a weighted combination of the regret value and the best increase.
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

    public Solution generateSolution(ProblemInstance instance, List<Integer> partialPath, Set<Integer> selected, int k) {
        // partialPath: currently constructed partial solution
        // selected: set of already selected nodes
        // k: target number of nodes in the solution
        Node[] nodes = instance.nodes;
        int n = nodes.length;
        int[][] distanceMatrix = instance.distanceMatrix;
        List<Integer> path = new ArrayList<>(partialPath);

        while (path.size() < k) {
            int bestNode = -1;
            double bestWeightedValue = Double.NEGATIVE_INFINITY;
            int bestInsertionPosition = -1;

            for (int node = 0; node < n; node++) {
                if (selected.contains(node)) {
                    continue;
                }
                InsertionInfo insertionInfo = findBestAndSecondBestInsertion(path, node, distanceMatrix);

                if (insertionInfo == null) {
                    continue;
                }

                int regretValue = insertionInfo.secondBestIncrease - insertionInfo.bestIncrease;
                double weightedValue = w1 * regretValue - w2 * (insertionInfo.bestIncrease + nodes[node].cost);

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
                // No more nodes can be added
                break;
            }
        }

        int objective = computeObjective(path.stream().mapToInt(Integer::intValue).toArray(), instance);
        return new Solution(path.stream().mapToInt(Integer::intValue).toArray(), objective);
    }

    protected InsertionInfo findBestAndSecondBestInsertion(List<Integer> path, int nodeToInsert, int[][] distanceMatrix) {
        if (path.isEmpty()) {
            return new InsertionInfo(0,0,0);
        }

        int bestIncrease = Integer.MAX_VALUE;
        int secondBestIncrease = Integer.MAX_VALUE;
        int bestPos = -1;
        int pathSize = path.size();

        for (int i = 0; i < pathSize; i++) {
            int current = path.get(i);
            int next = path.get((i + 1) % pathSize);
            int increase = distanceMatrix[current][nodeToInsert] + distanceMatrix[nodeToInsert][next] - distanceMatrix[current][next];

            if (increase < bestIncrease) {
                secondBestIncrease = bestIncrease;
                bestIncrease = increase;
                bestPos = i + 1;
            } else if (increase < secondBestIncrease) {
                secondBestIncrease = increase;
            }
        }

        // If we never found a second best, set it equal to best
        if (secondBestIncrease == Integer.MAX_VALUE) {
            secondBestIncrease = bestIncrease;
        }

        return new InsertionInfo(bestPos, bestIncrease, secondBestIncrease);
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
}

/**
 * Large Neighborhood Search (LNS) implementation.
 * The LNS algorithm:
 * 1. Generate an initial solution x (random)
 * 2. x := Local search(x) (optional)
 * 3. Repeat until time/iteration limits:
 *    a. y := Destroy(x) (remove a large fraction of nodes)
 *    b. y := Repair(y) using GreedyWeightedRegret
 *    c. y := Local search(y) (optional, depending on version)
 *    d. If f(y) > f(x) then x := y
 */
class LargeNeighborhoodSearch extends Heuristic {
    private final ProblemInstance instance;
    private final int k;
    private final double maxTimeMs;
    private final boolean applyLocalSearchEachIteration;
    private final double destroyFraction = 0.25; // remove about 20-30%
    private final GreedyWeightedRegret repairHeuristic;
    private final SteepestLocalSearchWithMoveEvaluation localSearch;

    public int numIterations;

    public LargeNeighborhoodSearch(ProblemInstance instance, int k, double maxTimeMs, boolean applyLocalSearchEachIteration) {
        this.instance = instance;
        this.k = k;
        this.maxTimeMs = maxTimeMs;
        this.applyLocalSearchEachIteration = applyLocalSearchEachIteration;
        this.repairHeuristic = new GreedyWeightedRegret();
        this.localSearch = new SteepestLocalSearchWithMoveEvaluation();
    }

    public Solution run() {
        long startTime = System.nanoTime();
        long maxDuration = (long) (maxTimeMs * 1e6); // Convert ms to ns

        // 1. Generate initial solution (random)
        int n = instance.nodes.length;
        int[] allNodes = new int[n];
        for (int i = 0; i < n; i++) {
            allNodes[i] = i;
        }
        shuffleArray(allNodes);
        int[] currentPath = Arrays.copyOfRange(allNodes, 0, k);
        shuffleArray(currentPath);

        // 2. Apply local search to initial solution
        Solution currentSol = localSearch.generateSolutionFromPath(instance, currentPath, Integer.MAX_VALUE);

        // Main loop
        numIterations = 0;
        while ((System.nanoTime() - startTime) < maxDuration) {
            numIterations++;

            // a. Destroy current solution
            Solution destroyed = destroy(currentSol);

            // b. Repair solution using GreedyWeightedRegret
            Solution repaired = repair(destroyed);

            // c. Local search (optional)
            if (applyLocalSearchEachIteration) {
                repaired = localSearch.generateSolutionFromPath(instance, repaired.path, Integer.MAX_VALUE);
            }

            // d. If better, accept
            if (repaired.objectiveValue < currentSol.objectiveValue) {
                currentSol = repaired;
            }
        }

        return currentSol;
    }

    private Solution destroy(Solution solution) {
        // Destroy operator: remove a fraction of nodes from solution
        // We'll remove about destroyFraction * k nodes randomly
        List<Integer> pathList = Arrays.stream(solution.path).boxed().collect(Collectors.toList());
        int removeCount = (int) Math.ceil(destroyFraction * pathList.size());

        // Remove random nodes (not the entire path, but chosen at random)
        for (int i = 0; i < removeCount; i++) {
            int removeIndex = random.nextInt(pathList.size());
            pathList.remove(removeIndex);
        }

        int[] newPath = pathList.stream().mapToInt(Integer::intValue).toArray();
        int newObjective = computeObjective(newPath, instance);
        return new Solution(newPath, newObjective);
    }

    private Solution repair(Solution partialSolution) {
        // Use the GreedyWeightedRegret to insert nodes until we have k nodes again
        Set<Integer> selected = Arrays.stream(partialSolution.path).boxed().collect(Collectors.toSet());
        List<Integer> partialPath = Arrays.stream(partialSolution.path).boxed().collect(Collectors.toList());

        return repairHeuristic.generateSolution(instance, partialPath, selected, k);
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

public class Main {
    public static void main(String[] args) {
        String inputDirPath = "inputs";
        File inputDir = new File(inputDirPath);

        if (!inputDir.exists() || !inputDir.isDirectory()) {
            System.err.println("Input directory '" + inputDirPath + "' does not exist or is not a directory.");
            return;
        }

        File[] inputFiles = inputDir.listFiles((_, name) -> name.toLowerCase().endsWith(".csv"));
        if (inputFiles != null) {
            Arrays.sort(inputFiles, Comparator.comparing(File::getName));
        }

        if (inputFiles == null || inputFiles.length == 0) {
            System.err.println("No CSV files found in the input directory.");
            return;
        }

        // According to the instructions, we use the average running time of MSLS from previous problem (~870 ms)
        // as the stopping criterion for LNS to allow fair comparison.
        double maxTimeMs = 870.0;

        // We will implement and test two versions of LNS:
        // 1. LNS with local search after each destroy-repair iteration
        // 2. LNS without local search after each destroy-repair iteration
        // Always apply local search to the initial solution inside the LNS.

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
                continue;
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

            // Run LNS with local search after each iteration
            System.out.println("Running LNS with local search after each iteration...");
            List<Solution> lnsWithLSsolutions = new ArrayList<>();
            List<Integer> lnsWithLSIterations = new ArrayList<>();
            double totalLnsWithLsTime = 0.0;

            for (int run = 0; run < 20; run++) {
                LargeNeighborhoodSearch lns = new LargeNeighborhoodSearch(instance, k, maxTimeMs, true);
                long startTime = System.nanoTime();
                Solution sol = lns.run();
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;
                totalLnsWithLsTime += durationMs;
                lnsWithLSsolutions.add(sol);
                lnsWithLSIterations.add(lns.numIterations);
            }

            double avgLnsWithLsTime = totalLnsWithLsTime / 20;
            double avgLnsWithLsIterations = lnsWithLSIterations.stream().mapToDouble(a -> a).average().orElse(0.0);

            Statistics lnsWithLSStats = Statistics.computeStatistics(lnsWithLSsolutions);

            // Run LNS without local search after each iteration
            System.out.println("Running LNS without local search after each iteration...");
            List<Solution> lnsNoLSsolutions = new ArrayList<>();
            List<Integer> lnsNoLSIterations = new ArrayList<>();
            double totalLnsNoLsTime = 0.0;

            for (int run = 0; run < 20; run++) {
                LargeNeighborhoodSearch lns = new LargeNeighborhoodSearch(instance, k, maxTimeMs, false);
                long startTime = System.nanoTime();
                Solution sol = lns.run();
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;
                totalLnsNoLsTime += durationMs;
                lnsNoLSsolutions.add(sol);
                lnsNoLSIterations.add(lns.numIterations);
            }

            double avgLnsNoLsTime = totalLnsNoLsTime / 20;
            double avgLnsNoLsIterations = lnsNoLSIterations.stream().mapToDouble(a -> a).average().orElse(0.0);

            Statistics lnsNoLSStats = Statistics.computeStatistics(lnsNoLSsolutions);

            // Output results
            System.out.println("\n--- LNS Computational Results for Instance: " + instanceName + " ---\n");

            // Results for LNS with LS
            System.out.println("Method: LNS with local search after each iteration");
            System.out.println("Min Objective: " + lnsWithLSStats.minObjective);
            System.out.println("Max Objective: " + lnsWithLSStats.maxObjective);
            System.out.printf("Average Objective: %.2f%n", lnsWithLSStats.avgObjective);
            System.out.printf("Average Time per run: %.2f ms%n", avgLnsWithLsTime);
            System.out.printf("Average Number of Iterations: %.2f%n", avgLnsWithLsIterations);
            System.out.println("Best Solution Path: " + Arrays.toString(lnsWithLSStats.bestPath) + "\n");

            // Save best path for LNS with LS
            String lnsWithLSFileName = outputInstanceDirPath + "/LNS_with_LS.csv";
            try {
                saveBestPathToCSV(lnsWithLSStats.bestPath, lnsWithLSFileName);
                System.out.println("Best path for LNS with LS saved to " + lnsWithLSFileName + "\n");
            } catch (IOException e) {
                System.err.println("Error writing best path to CSV for LNS with LS: " + e.getMessage());
            }

            // Results for LNS without LS
            System.out.println("Method: LNS without local search after each iteration");
            System.out.println("Min Objective: " + lnsNoLSStats.minObjective);
            System.out.println("Max Objective: " + lnsNoLSStats.maxObjective);
            System.out.printf("Average Objective: %.2f%n", lnsNoLSStats.avgObjective);
            System.out.printf("Average Time per run: %.2f ms%n", avgLnsNoLsTime);
            System.out.printf("Average Number of Iterations: %.2f%n", avgLnsNoLsIterations);
            System.out.println("Best Solution Path: " + Arrays.toString(lnsNoLSStats.bestPath) + "\n");

            // Save best path for LNS without LS
            String lnsNoLSFileName = outputInstanceDirPath + "/LNS_no_LS.csv";
            try {
                saveBestPathToCSV(lnsNoLSStats.bestPath, lnsNoLSFileName);
                System.out.println("Best path for LNS without LS saved to " + lnsNoLSFileName + "\n");
            } catch (IOException e) {
                System.err.println("Error writing best path to CSV for LNS without LS: " + e.getMessage());
            }

            System.out.println("Finished processing instance: " + instanceName + "\n");
        }

        // Attempt to run Python script for plotting (if applicable)
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