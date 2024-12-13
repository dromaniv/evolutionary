import java.io.*;
import java.util.*;
import java.util.Comparator;
import java.util.stream.Collectors;

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
 * Implements Local Search algorithms with specified options.
 */
class LocalSearch extends Heuristic {
    enum LocalSearchType { STEEPEST, GREEDY }

    enum IntraRouteMoveType { TWO_NODES_EXCHANGE, TWO_EDGES_EXCHANGE }

    enum MoveType { INTRA_ROUTE, INTER_ROUTE }

    private static class Move {
        MoveType moveType;
        int i, j; // Positions in path for intra-route moves
        int u; // Unselected node for inter-route moves
        IntraRouteMoveType intraRouteMoveType;

        public Move(MoveType moveType, int i, int j, IntraRouteMoveType intraRouteMoveType) {
            this.moveType = moveType;
            this.i = i;
            this.j = j;
            this.intraRouteMoveType = intraRouteMoveType;
        }

        public Move(MoveType moveType, int i, int u) {
            this.moveType = moveType;
            this.i = i;
            this.u = u;
        }
    }

    /**
     * Performs local search on the given initial solution.
     * @param initialSolution The starting solution
     * @param instance The problem instance
     * @param localSearchType Type of local search (STEEPEST or GREEDY)
     * @param intraRouteMoveType Type of intra-route moves
     * @param isGreedy Whether the neighborhood should be browsed in random order
     * @return An improved Solution object
     */
    public Solution performLocalSearch(
            Solution initialSolution,
            ProblemInstance instance,
            LocalSearchType localSearchType,
            IntraRouteMoveType intraRouteMoveType,
            boolean isGreedy
    ) {
        List<Integer> path = new ArrayList<>(initialSolution.getPath());
        Set<Integer> selectedNodes = new HashSet<>(path);
        int n = instance.getNodes().size();
        int[][] distanceMatrix = instance.getDistanceMatrix();
        List<Node> nodes = instance.getNodes();

        boolean improvement = true;
        while (improvement) {
            improvement = false;
            Move bestMove = null;
            int bestDelta = 0;

            List<Move> moves = generateMoves(path, selectedNodes, n, intraRouteMoveType);

            if (isGreedy) {
                Collections.shuffle(moves, random);
            }

            for (Move move : moves) {
                int delta = computeDelta(move, path, selectedNodes, distanceMatrix, nodes);

                if (delta < 0) {
                    if (localSearchType == LocalSearchType.GREEDY) {
                        // Apply the move
                        applyMove(move, path, selectedNodes);
                        improvement = true;
                        break;
                    } else if (localSearchType == LocalSearchType.STEEPEST) {
                        if (delta < bestDelta) {
                            bestDelta = delta;
                            bestMove = move;
                        }
                    }
                }
            }

            if (!improvement && localSearchType == LocalSearchType.STEEPEST && bestMove != null) {
                // Apply the best move
                applyMove(bestMove, path, selectedNodes);
                improvement = true;
            }
        }

        // Recompute objective value
        int newObjective = computeObjective(path, instance);
        return new Solution(path, newObjective);
    }

    private List<Move> generateMoves(List<Integer> path, Set<Integer> selectedNodes, int n, IntraRouteMoveType intraRouteMoveType) {
        List<Move> moves = new ArrayList<>();

        // Generate intra-route moves
        int pathSize = path.size();
        for (int i = 0; i < pathSize - 1; i++) {
            for (int j = i + 1; j < pathSize; j++) {
                if (intraRouteMoveType == IntraRouteMoveType.TWO_EDGES_EXCHANGE && (i == j || (i + 1) % pathSize == j || i == (j + 1) % pathSize)) {
                    continue; // Skip adjacent nodes for two-edges exchange
                }
                moves.add(new Move(MoveType.INTRA_ROUTE, i, j, intraRouteMoveType));
            }
        }

        // Generate inter-route moves
        for (int i = 0; i < pathSize; i++) {
            for (int u = 0; u < n; u++) {
                if (!selectedNodes.contains(u)) {
                    moves.add(new Move(MoveType.INTER_ROUTE, i, u));
                }
            }
        }

        return moves;
    }

    private int computeDelta(Move move, List<Integer> path, Set<Integer> selectedNodes, int[][] distanceMatrix, List<Node> nodes) {
        int delta = 0;
        if (move.moveType == MoveType.INTRA_ROUTE) {
            if (move.intraRouteMoveType == IntraRouteMoveType.TWO_NODES_EXCHANGE) {
                delta = computeDeltaTwoNodesExchange(move.i, move.j, path, distanceMatrix);
            } else if (move.intraRouteMoveType == IntraRouteMoveType.TWO_EDGES_EXCHANGE) {
                delta = computeDeltaTwoEdgesExchange(move.i, move.j, path, distanceMatrix);
            }
        } else if (move.moveType == MoveType.INTER_ROUTE) {
            delta = computeDeltaInterRouteExchange(move.i, move.u, path, selectedNodes, distanceMatrix, nodes);
        }
        return delta;
    }

    private int computeDeltaTwoNodesExchange(int i, int j, List<Integer> path, int[][] distanceMatrix) {
        int n = path.size();

        int node_i_prev = path.get((i - 1 + n) % n);
        int node_i = path.get(i);
        int node_i_next = path.get((i + 1) % n);

        int node_j_prev = path.get((j - 1 + n) % n);
        int node_j = path.get(j);
        int node_j_next = path.get((j + 1) % n);

        int delta;

        if ((i + 1) % n == j) { // i and j are adjacent, i before j
            int distBefore = distanceMatrix[node_i_prev][node_i] + distanceMatrix[node_i][node_j] + distanceMatrix[node_j][node_j_next];
            int distAfter = distanceMatrix[node_i_prev][node_j] + distanceMatrix[node_j][node_i] + distanceMatrix[node_i][node_j_next];

            delta = distAfter - distBefore;
        } else if ((j + 1) % n == i) { // j and i are adjacent, j before i
            int distBefore = distanceMatrix[node_j_prev][node_j] + distanceMatrix[node_j][node_i] + distanceMatrix[node_i][node_i_next];
            int distAfter = distanceMatrix[node_j_prev][node_i] + distanceMatrix[node_i][node_j] + distanceMatrix[node_j][node_i_next];

            delta = distAfter - distBefore;
        } else {
            int distBefore = distanceMatrix[node_i_prev][node_i] + distanceMatrix[node_i][node_i_next]
                    + distanceMatrix[node_j_prev][node_j] + distanceMatrix[node_j][node_j_next];

            int distAfter = distanceMatrix[node_i_prev][node_j] + distanceMatrix[node_j][node_i_next]
                    + distanceMatrix[node_j_prev][node_i] + distanceMatrix[node_i][node_j_next];

            delta = distAfter - distBefore;
        }

        return delta;
    }

    private int computeDeltaTwoEdgesExchange(int i, int j, List<Integer> path, int[][] distanceMatrix) {
        int n = path.size();
        if (i == j || (i + 1) % n == j || i == (j + 1) % n) {
            // No change for adjacent nodes
            return 0;
        }

        int i1 = Math.min(i, j);
        int j1 = Math.max(i, j);

        int node_i_prev = path.get((i1 - 1 + n) % n);
        int node_i = path.get(i1);
        int node_j = path.get(j1);
        int node_j_next = path.get((j1 + 1) % n);

        int delta = 0;

        // Edges being removed
        delta -= distanceMatrix[node_i_prev][node_i];
        delta -= distanceMatrix[node_j][node_j_next];

        // Edges being added
        delta += distanceMatrix[node_i_prev][node_j];
        delta += distanceMatrix[node_i][node_j_next];

        return delta;
    }

    private int computeDeltaInterRouteExchange(int i, int u, List<Integer> path, Set<Integer> selectedNodes, int[][] distanceMatrix, List<Node> nodes) {
        int n = path.size();
        int node_i_prev = path.get((i - 1 + n) % n);
        int node_i = path.get(i);
        int node_i_next = path.get((i + 1) % n);

        int delta_distance = distanceMatrix[node_i_prev][u] + distanceMatrix[u][node_i_next]
                - distanceMatrix[node_i_prev][node_i] - distanceMatrix[node_i][node_i_next];

        int delta_cost = nodes.get(u).getCost() - nodes.get(node_i).getCost();

        int delta = delta_distance + delta_cost;

        return delta;
    }

    private void applyMove(Move move, List<Integer> path, Set<Integer> selectedNodes) {
        if (move.moveType == MoveType.INTRA_ROUTE) {
            if (move.intraRouteMoveType == IntraRouteMoveType.TWO_NODES_EXCHANGE) {
                Collections.swap(path, move.i, move.j);
            } else if (move.intraRouteMoveType == IntraRouteMoveType.TWO_EDGES_EXCHANGE) {
                int i = Math.min(move.i, move.j);
                int j = Math.max(move.i, move.j);
                while (i < j) {
                    Collections.swap(path, i, j);
                    i++;
                    j--;
                }
            }
        } else if (move.moveType == MoveType.INTER_ROUTE) {
            int oldNode = path.get(move.i);
            path.set(move.i, move.u);
            selectedNodes.remove(oldNode);
            selectedNodes.add(move.u);
        }
    }

    @Override
    public Solution generateSolution(ProblemInstance instance, int k, int startNode) {
        // This method is not used for LocalSearch; implementations are in performLocalSearch
        return null;
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
    private static void saveAllPathsToCSV(List<Solution> solutions, String fileName) throws IOException {
        // Sort the solutions based on objective value in ascending order
        solutions.sort(Comparator.comparingInt(Solution::getObjectiveValue));
    
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        for (Solution sol : solutions) {
            int objectiveValue = sol.getObjectiveValue();
            List<Integer> path = sol.getPath();
    
            // Convert path to comma-separated string
            String pathString = path.stream()
                    .map(String::valueOf)
                    .collect(Collectors.joining(", "));
    
            // Write to file in the format: objective; path
            writer.write(objectiveValue + "; " + pathString);
            writer.newLine();
        }
        writer.close();
    }
    public static void main(String[] args) {
        // Define the input directory
        String inputDirPath = "C:\\Users\\wazus\\OneDrive\\Desktop\\Evolutionary Computation ALL\\evolutionary\\inputs";
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
        Heuristic randomHeuristic = new RandomSolution();
        Heuristic greedyHeuristic = new GreedyWeightedRegret();
        LocalSearch localSearch = new LocalSearch();

        // Define combinations of options
        LocalSearch.LocalSearchType[] searchTypes = {LocalSearch.LocalSearchType.GREEDY};
        LocalSearch.IntraRouteMoveType[] moveTypes = {LocalSearch.IntraRouteMoveType.TWO_EDGES_EXCHANGE};
        boolean[] startingSolutions = {true}; // true for random, false for greedy

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
            String outputInstanceDirPath = "C:\\Users\\wazus\\OneDrive\\Desktop\\Evolutionary Computation ALL\\outputs\\" + instanceName;
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

            // Generate combinations
            for (LocalSearch.LocalSearchType searchType : searchTypes) {
                for (LocalSearch.IntraRouteMoveType moveType : moveTypes) {
                    for (boolean isRandomStart : startingSolutions) {
                        String methodName = searchType + "_" + moveType + "_" + "RandomStart";
                        methodSolutions.put(methodName, new ArrayList<>());

                        System.out.println("Running method: " + methodName);
                        List<Solution> solutions = new ArrayList<>();

                        // Start timing
                        long startTime = System.nanoTime();

                        if (isRandomStart) {
                            // Use 200 random starting solutions
                            for (int run = 0; run < 1000; run++) {
                                Solution initialSolution = randomHeuristic.generateSolution(instance, k, -1);
                                Solution improvedSolution = localSearch.performLocalSearch(
                                        initialSolution, instance, searchType, moveType, searchType == LocalSearch.LocalSearchType.GREEDY);
                                solutions.add(improvedSolution);
                            }
                        } else {
                            // Use each node as starting node for greedy heuristic
                            for (int startNode = 0; startNode < n && startNode < 1000; startNode++) {
                                Solution initialSolution = greedyHeuristic.generateSolution(instance, k, startNode);
                                Solution improvedSolution = localSearch.performLocalSearch(
                                        initialSolution, instance, searchType, moveType, searchType == LocalSearch.LocalSearchType.GREEDY);
                                solutions.add(improvedSolution);
                            }
                        }

                        // End timing
                        long endTime = System.nanoTime();
                        long duration = endTime - startTime;
                        double durationMs = duration / 1e6;

                        methodSolutions.get(methodName).addAll(solutions);
                        methodTimes.put(methodName, durationMs);
                    }
                }
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
                String outputFileName = outputInstanceDirPath + "/" + methodName + "_all_solutions.csv";
                try {
                    saveAllPathsToCSV(solutions, outputFileName);
                    System.out.println("All solutions saved to " + outputFileName + "\n");
                } catch (IOException e) {
                    System.err.println("Error writing solutions to CSV for method '" + methodName + "': " + e.getMessage());
                }
            }
            System.out.println("Finished processing instance: " + instanceName + "\n");
        }
    }
}
