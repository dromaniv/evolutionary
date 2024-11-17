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
    final Random random = new Random(90909090);

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
        Node[] nodes = instance.nodes;
        int n = nodes.length;
        int[] allNodes = new int[n];
        for (int i = 0; i < n; i++) {
            allNodes[i] = i;
        }

        // Start with a random solution
        int[] currentPath = Arrays.copyOfRange(allNodes, 0, k);
        shuffleArray(currentPath);

        int[][] distanceMatrix = instance.distanceMatrix;
        boolean[] inPathSet = new boolean[n];
        for (int node : currentPath) {
            inPathSet[node] = true;
        }

        boolean improvement = true;
        List<Move> improvingMoves = new ArrayList<>();

        while (improvement) {
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
                System.out.println("Min Objective: " + stats.minObjective);
                System.out.println("Max Objective: " + stats.maxObjective);
                System.out.printf("Average Objective: %.2f%n", stats.avgObjective);
                System.out.printf("Time taken: %.2f ms%n", methodTimes.get(methodName));
                System.out.println("Best Solution Path: " + Arrays.toString(stats.bestPath) + "\n");

                // Save best path to CSV
                String outputFileName = outputInstanceDirPath + "/" + methodName + ".csv";
                try {
                    saveBestPathToCSV(stats.bestPath, outputFileName);
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