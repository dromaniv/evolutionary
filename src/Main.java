import java.io.*;
import java.util.*;
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
 * Represents a problem instance with nodes, a distance matrix, and candidate edges.
 */
class ProblemInstance {
    private final List<Node> nodes;
    private int[][] distanceMatrix;
    private List<List<Integer>> candidateEdges; // For each node, its 10 nearest neighbors

    public ProblemInstance() {
        nodes = new ArrayList<>();
    }

    /**
     * Reads nodes from a CSV file. Each line should have x;y;cost
     *
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
     * Computes the candidate edges for each node based on the sum of edge length and node cost.
     *
     * @param numCandidates Number of nearest neighbors to consider for each node
     */
    public void computeCandidateEdges(int numCandidates) {
        int n = nodes.size();
        candidateEdges = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            List<Integer> neighbors = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                neighbors.add(j);
            }
            final int currentI = i; // Make a final copy for lambda
            neighbors.sort(Comparator.comparingInt(j -> distanceMatrix[currentI][j] + nodes.get(j).getCost()));
            List<Integer> topNeighbors = neighbors.stream().limit(numCandidates).collect(Collectors.toList());
            candidateEdges.add(topNeighbors);
        }
    }

    public List<Node> getNodes() {
        return nodes;
    }

    public int[][] getDistanceMatrix() {
        return distanceMatrix;
    }

    public List<List<Integer>> getCandidateEdges() {
        return candidateEdges;
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
     *
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
 * Implements the Steepest Local Search with Candidate Moves.
 */
/**
 * Implements the Steepest Local Search with Candidate Moves.
 */
class LocalSearch {
    private List<List<Integer>> candidateEdges; // Accessed from ProblemInstance

    /**
     * Represents a potential move with its delta objective.
     */
    private static class PotentialMove {
        MoveType moveType;
        int pos1;
        int pos2; // For intra-route moves
        int node;  // For inter-route moves
        int delta;

        public PotentialMove(MoveType moveType, int pos1, int pos2, int node, int delta) {
            this.moveType = moveType;
            this.pos1 = pos1;
            this.pos2 = pos2;
            this.node = node;
            this.delta = delta;
        }
    }

        /**
     * Computes the objective function: sum of path lengths + sum of node costs
     *
     * @param path           The current path
     * @param distanceMatrix The distance matrix
     * @param nodes          The list of nodes
     * @return The objective value
     */
    private int computeObjective(List<Integer> path, int[][] distanceMatrix, List<Node> nodes) {
        int totalDistance = 0;
        int k = path.size();
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

    private enum MoveType { INTRA_ROUTE_TWO_EDGES_EXCHANGE, INTER_ROUTE_NODE_SWAP }

    /**
     * Performs the steepest local search with candidate moves on the given initial solution.
     *
     * @param initialSolution The starting solution
     * @param instance        The problem instance
     * @return An improved Solution object
     */
    public Solution performSteepestLocalSearch(Solution initialSolution, ProblemInstance instance) {
        // Initialize
        List<Integer> path = new ArrayList<>(initialSolution.getPath());
        int[][] distanceMatrix = instance.getDistanceMatrix();
        List<Node> nodes = instance.getNodes();
        this.candidateEdges = instance.getCandidateEdges();
        Set<Integer> selectedNodes = new HashSet<>(path);

        boolean improvement = true;
        while (improvement) {
            improvement = false;
            PotentialMove bestMove = null;

            List<PotentialMove> allPotentialMoves = new ArrayList<>();

            // Generate all possible intra-route two-edges exchanges that introduce at least one candidate edge
            int n = path.size();
            for (int i = 0; i < n; i++) {
                for (int j = i + 2; j < n; j++) { // Ensure that we are not swapping adjacent edges
                    // Calculate the potential delta
                    int iPrev = path.get((i - 1 + n) % n);
                    int iNode = path.get(i);
                    int jNode = path.get(j);
                    int jNext = path.get((j + 1) % n);

                    // Current edges: (iPrev - iNode), (jNode - jNext)
                    // New edges: (iPrev - jNode), (iNode - jNext)
                    int currentDistance = distanceMatrix[iPrev][iNode] + distanceMatrix[jNode][jNext];
                    int newDistance = distanceMatrix[iPrev][jNode] + distanceMatrix[iNode][jNext];
                    int delta = newDistance - currentDistance;

                    // Check if at least one new edge is a candidate edge
                    boolean introducesCandidateEdge = false;
                    if (candidateEdges.get(iPrev).contains(jNode) || candidateEdges.get(jNode).contains(iPrev) ||
                        candidateEdges.get(iNode).contains(jNext) || candidateEdges.get(jNext).contains(iNode)) {
                        introducesCandidateEdge = true;
                    }

                    if (introducesCandidateEdge) {
                        allPotentialMoves.add(new PotentialMove(MoveType.INTRA_ROUTE_TWO_EDGES_EXCHANGE, i, j, -1, delta));
                    }
                }
            }

            // Generate all possible inter-route node swaps that introduce at least one candidate edge
            for (int i = 0; i < n; i++) {
                int selectedNode = path.get(i);
                for (int u = 0; u < nodes.size(); u++) {
                    if (selectedNodes.contains(u)) continue; // Skip already selected nodes

                    /*
                        Swap selectedNode with u:
                        - Remove selectedNode: (prev - selectedNode - next) becomes (prev - next)
                        - Add u: (prev - u - next)
                        Delta distance: (distanceMatrix[prev][u] + distanceMatrix[u][next]) - (distanceMatrix[prev][selectedNode] + distanceMatrix[selectedNode][next])
                        Delta cost: cost[u] - cost[selectedNode]
                    */
                    int prev = path.get((i - 1 + n) % n);
                    int next = path.get((i + 1) % n);
                    int deltaDistance = distanceMatrix[prev][u] + distanceMatrix[u][next] - distanceMatrix[prev][selectedNode] - distanceMatrix[selectedNode][next];
                    int deltaCost = nodes.get(u).getCost() - nodes.get(selectedNode).getCost();
                    int deltaTotal = deltaDistance + deltaCost;

                    // Check if introducing u introduces at least one candidate edge
                    boolean introducesCandidateEdge = false;
                    if (candidateEdges.get(prev).contains(u) || candidateEdges.get(u).contains(prev) ||
                        candidateEdges.get(u).contains(next) || candidateEdges.get(next).contains(u)) {
                        introducesCandidateEdge = true;
                    }

                    if (introducesCandidateEdge) {
                        allPotentialMoves.add(new PotentialMove(MoveType.INTER_ROUTE_NODE_SWAP, i, -1, u, deltaTotal));
                    }
                }
            }

            // Find the best move (steepest)
            for (PotentialMove move : allPotentialMoves) {
                if (bestMove == null || move.delta < bestMove.delta) {
                    bestMove = move;
                }
            }

            // Apply the best move if it improves the solution
            if (bestMove != null && bestMove.delta < 0) {
                improvement = true;
                if (bestMove.moveType == MoveType.INTRA_ROUTE_TWO_EDGES_EXCHANGE) {
                    // Perform two-edges exchange between positions bestMove.pos1 and bestMove.pos2
                    int pos1 = bestMove.pos1;
                    int pos2 = bestMove.pos2;
                    Collections.swap(path, pos1, pos2);
                } else if (bestMove.moveType == MoveType.INTER_ROUTE_NODE_SWAP) {
                    // Swap selected node at position bestMove.pos1 with node 'bestMove.node'
                    int pos = bestMove.pos1;
                    int oldNode = path.get(pos); // Capture the old node before swapping
                    path.set(pos, bestMove.node);
                    selectedNodes.remove(oldNode); // Remove the old node from selectedNodes
                    selectedNodes.add(bestMove.node); // Add the new node to selectedNodes
                }
            }
        }

        // Recompute objective value
        int newObjective = computeObjective(path, distanceMatrix, nodes);
        return new Solution(path, newObjective);
    }
}

/**
 * The main class to execute the program.
 */
public class Main {
    public static void main(String[] args) {
        // Define the input directory
        String inputDirPath = "C:\\Users\\wazus\\OneDrive\\Desktop\\Evolutionary 3\\evolutionary\\inputs";
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

        // Initialize local search
        LocalSearch localSearch = new LocalSearch();

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

                System.out.println("Computing candidate edges...");
                instance.computeCandidateEdges(10); // 10 nearest neighbors
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

            // Initialize a list to store all solutions
            List<Solution> allSolutions = new ArrayList<>();

            // Initialize the distance matrix and nodes
            int[][] distanceMatrix = instance.getDistanceMatrix();
            List<Node> nodes = instance.getNodes();

            // Generate 200 random starting solutions
            System.out.println("Generating 200 random starting solutions and performing local search...");

            // Initialize random
            Random random = new Random();

            long startTime = System.nanoTime();

            for (int run = 0; run < 200; run++) {
                // Generate a random starting solution
                System.out.println("Run " + (run + 200) + " / 1");
                List<Integer> selectedNodes = generateRandomSolution(n, k, random);
                //System.out.println("Selected nodes: " + selectedNodes);
                int objective = computeObjective(selectedNodes, distanceMatrix, nodes);
                //System.out.println("Initial objective: " + objective);
                Solution initialSolution = new Solution(selectedNodes, objective);
                //System.out.println("Initial path: " + initialSolution.getPath());

                // Perform steepest local search with candidate moves
                Solution improvedSolution = localSearch.performSteepestLocalSearch(initialSolution, instance);
                //System.out.println("Improved objective: " + improvedSolution.getObjectiveValue());
                allSolutions.add(improvedSolution);
                //System.out.println("Improved path: " + improvedSolution.getPath() + "\n");
            }

            long endTime = System.nanoTime();
            double durationMs = (endTime - startTime) / 1e6;
            System.out.printf("Local search completed in %.2f ms.%n%n", durationMs);

            // Compute statistics
            Statistics stats = Statistics.computeStatistics(allSolutions);
            System.out.println("--- Computational Experiment Results for Instance: " + instanceName + " ---\n");
            System.out.println("Min Objective: " + stats.getMinObjective());
            System.out.println("Max Objective: " + stats.getMaxObjective());
            System.out.printf("Average Objective: %.2f%n", stats.getAvgObjective());
            System.out.println("Best Solution Path: " + stats.getBestPath() + "\n");

            // Save best path to CSV
            String outputFileName = outputInstanceDirPath + "/BestSolution.csv";
            try {
                saveBestPathToCSV(stats.getBestPath(), outputFileName);
                System.out.println("Best path saved to " + outputFileName + "\n");
            } catch (IOException e) {
                System.err.println("Error writing best path to CSV for instance '" + instanceName + "': " + e.getMessage());
            }

            System.out.println("Finished processing instance: " + instanceName + "\n");
        }

        // After processing all instances, run the Python script
        String pythonScript = "C:\\Users\\wazus\\OneDrive\\Desktop\\Evolutionary 3\\evolutionary\\plot_results.py"; // Adjust the path if necessary
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
     * Generates a random solution by selecting k unique nodes.
     *
     * @param n      Total number of nodes
     * @param k      Number of nodes to select
     * @param random Random instance
     * @return A list of selected node indices
     */
    private static List<Integer> generateRandomSolution(int n, int k, Random random) {
        List<Integer> allNodes = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            allNodes.add(i);
        }
        Collections.shuffle(allNodes, random);
        return allNodes.subList(0, Math.min(k, allNodes.size()));
    }

    /**
     * Computes the objective function: sum of path lengths + sum of node costs
     *
     * @param path           The current path
     * @param distanceMatrix The distance matrix
     * @param nodes          The list of nodes
     * @return The objective value
     */
    private static int computeObjective(List<Integer> path, int[][] distanceMatrix, List<Node> nodes) {
        int totalDistance = 0;
        int k = path.size();
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

    /**
     * Saves the best path to a CSV file, with each node index on a separate line.
     * The first node is appended at the end to complete the cycle.
     *
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
            writer.write(bestPath.get(0).toString()); // Complete the cycle
        }
        writer.close();
    }
}
