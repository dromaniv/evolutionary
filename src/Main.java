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
        for (int i = 0; i < n; i++) {
            Node nodeI = nodes[i];
            for (int j = i + 1; j < n; j++) {
                Node nodeJ = nodes[j];
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
 * Simple abstract class for heuristics: includes objective calculation.
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

        for (int i = 0; i < k; i++) {
            int from = path[i];
            int to = path[(i + 1) % k];
            totalDistance += distanceMatrix[from][to];
        }

        int totalCost = 0;
        for (int nodeIndex : path) {
            totalCost += nodes[nodeIndex].cost;
        }
        return totalDistance + totalCost;
    }
}

/**
 * A single "groundbreakingly better" class: Adaptive Large Neighborhood Search (ALNS)
 * with integrated local search and partial global memory.
 */
class AdaptiveLargeNeighborhoodSearch extends Heuristic {

    private final int maxNoImprovement;   // # of iterations allowed with no improvement before diversification
    private final double removalFraction; // fraction of nodes to remove each iteration
    private final int[] operatorScores;   // success score for each removal operator
    private final int[] operatorAttempts; // usage count for each removal operator
    private final List<RemovalOperator> removalOps; // available destruction ops

    /**
     * Constructor. You can tune the fraction removed or no-improvement threshold.
     */
    public AdaptiveLargeNeighborhoodSearch(int maxNoImprovement, double removalFraction) {
        this.maxNoImprovement = maxNoImprovement;
        this.removalFraction = removalFraction;

        // We'll define multiple removal operators below
        removalOps = new ArrayList<>();
        removalOps.add(new RandomRemoval());
        removalOps.add(new DistanceRemoval());
        removalOps.add(new CostRemoval());

        // Initialize scores and attempts for each operator
        operatorScores = new int[removalOps.size()];
        operatorAttempts = new int[removalOps.size()];
    }

    /**
     * Runs ALNS within a given time limit (ms). We'll do multiple iterations, adaptively
     * choosing removal operators, doing a "repair," local search, and updating the best solution.
     */
    public Solution run(ProblemInstance instance, int k, double maxTimeMs) {
        long startTime = System.nanoTime();
        long maxDuration = (long) (maxTimeMs * 1e6);

        // 1) Create an initial solution at random (k selected nodes)
        int[] initPath = generateRandomPath(k, instance.nodes.length);
        int initObj = computeObjective(initPath, instance);
        Solution bestSol = new Solution(initPath, initObj);

        // 2) Keep a current working solution
        int[] currPath = Arrays.copyOf(initPath, initPath.length);
        int currObj = initObj;

        // 3) To track no-improvement stretch
        int iterSinceImprovement = 0;

        // 4) Main loop until time is up
        while (System.nanoTime() - startTime < maxDuration) {
            // (a) Pick a removal operator using a roulette-wheel selection
            RemovalOperator chosenRemovalOp = selectRemovalOperator();

            // (b) Destroy: remove a fraction of nodes from currPath
            int[] destroyedPath = removeNodes(currPath, instance, chosenRemovalOp);

            // (c) Repair: re-insert them using Weighted Regret
            int[] repairedPath = repairPath(destroyedPath, instance, k);

            // (d) Local Search intensification
            int[] improvedPath = localSearch(repairedPath, instance);
            int improvedObj = computeObjective(improvedPath, instance);

            // (e) Evaluate
            operatorAttempts[removalOps.indexOf(chosenRemovalOp)]++;
            if (improvedObj < currObj) {
                // accept new current solution
                currPath = improvedPath;
                currObj = improvedObj;

                // give some reward to chosen operator
                operatorScores[removalOps.indexOf(chosenRemovalOp)] += 3; // e.g. reward 3 for improvement
            }

            // (f) If better than global best, record
            if (currObj < bestSol.objectiveValue) {
                bestSol = new Solution(currPath, currObj);
                iterSinceImprovement = 0;
            } else {
                iterSinceImprovement++;
            }

            // (g) If too long without improvement, do a random shuffle of curr
            if (iterSinceImprovement > maxNoImprovement) {
                randomShuffle(currPath);
                currObj = computeObjective(currPath, instance);
                iterSinceImprovement = 0;
            }
        }

        return bestSol;
    }

    /**
     * Selects a removal operator using weighted probabilities based on success scores.
     */
    private RemovalOperator selectRemovalOperator() {
        double[] probabilities = new double[removalOps.size()];
        double sumScores = 0.0;
        for (int i = 0; i < removalOps.size(); i++) {
            double ratio = (operatorAttempts[i] == 0) ? 1.0 : ((double) operatorScores[i] / operatorAttempts[i]);
            probabilities[i] = ratio;
            sumScores += ratio;
        }
        // normalize
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sumScores;
        }
        // random selection
        double r = random.nextDouble();
        double cum = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cum += probabilities[i];
            if (r <= cum) {
                return removalOps.get(i);
            }
        }
        return removalOps.get(probabilities.length - 1); // fallback
    }

    /**
     * Removes 'removalFraction * path.length' nodes from the path using chosen removal operator
     */
    private int[] removeNodes(int[] path, ProblemInstance instance, RemovalOperator op) {
        // Make a copy of the path as a list to remove from
        List<Integer> nodeList = Arrays.stream(path).boxed().collect(Collectors.toList());
        int removalCount = (int) Math.max(1, Math.floor(removalFraction * nodeList.size()));
        // We'll track removed nodes in a separate list
        List<Integer> removed = op.selectNodesToRemove(nodeList, removalCount, instance);
        // Actually remove them
        nodeList.removeAll(removed);
        return nodeList.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Repairs a partially destroyed solution (the 'destroyedPath') back to 'k' nodes
     * using Weighted Regret Insertion, reusing the approach from your previous code.
     */
    private int[] repairPath(int[] destroyedPath, ProblemInstance instance, int k) {
        GreedyWeightedRegret regret = new GreedyWeightedRegret();
        List<Integer> partialPath = Arrays.stream(destroyedPath).boxed().collect(Collectors.toList());
        Set<Integer> selected = new HashSet<>(partialPath);
        return regret.generateSolution(instance, partialPath, selected, k).path;
    }

    /**
     * Applies local search to the repaired path. Uses 2-opt and single-node swap for intensification.
     */
    private int[] localSearch(int[] path, ProblemInstance instance) {
        boolean improved = true;
        int[] bestPath = Arrays.copyOf(path, path.length);
        int bestObj = computeObjective(bestPath, instance);

        while (improved) {
            improved = false;

            // 1) 2-opt moves
            for (int i = 0; i < bestPath.length; i++) {
                for (int j = i + 2; j < bestPath.length; j++) {
                    if (i == 0 && j == bestPath.length - 1) {
                        continue; // do not reverse entire route
                    }
                    int delta = evaluate2OptDelta(instance, bestPath, i, j);
                    if (delta < 0) {
                        apply2Opt(bestPath, i, j);
                        bestObj += delta;
                        improved = true;
                    }
                }
            }

            // 2) Single-node swap with outside (like inter-route)
            // But we must pick a node from the path to swap with a node not in the path
            // We'll do a few random attempts
            Set<Integer> inPath = Arrays.stream(bestPath).boxed().collect(Collectors.toSet());
            int tries = 0;
            while (tries < 10) {
                tries++;
                int pos = random.nextInt(bestPath.length);
                int oldNode = bestPath[pos];
                int newNode = random.nextInt(instance.nodes.length);
                if (inPath.contains(newNode)) {
                    continue;
                }
                int delta = evaluateNodeSwapDelta(instance, bestPath, pos, newNode);
                if (delta < 0) {
                    bestPath[pos] = newNode;
                    inPath.remove(oldNode);
                    inPath.add(newNode);
                    bestObj += delta;
                    improved = true;
                }
            }
        }
        return bestPath;
    }

    /**
     * Evaluates the delta in cost for a 2-opt on edges (i, i+1) and (j, j+1).
     */
    private int evaluate2OptDelta(ProblemInstance instance, int[] path, int i, int j) {
        int[][] dist = instance.distanceMatrix;
        int n = path.length;
        int a = path[i];
        int b = path[(i + 1) % n];
        int c = path[j];
        int d = path[(j + 1) % n];
        return -dist[a][b] - dist[c][d] + dist[a][c] + dist[b][d];
    }

    private void apply2Opt(int[] path, int i, int j) {
        int n = path.length;
        // Reverse segment path[i+1..j]
        int start = i+1, end = j;
        while (start < end) {
            int temp = path[start];
            path[start] = path[end];
            path[end] = temp;
            start++;
            end--;
        }
    }

    /**
     * Evaluates the cost difference if we swap the node at position 'pos' with 'newNode' not in the path.
     * We'll remove oldNode from edges (pos-1, pos), (pos, pos+1) and add newNode in that position.
     */
    private int evaluateNodeSwapDelta(ProblemInstance instance, int[] path, int pos, int newNode) {
        int[][] dist = instance.distanceMatrix;
        Node[] nodes = instance.nodes;
        int oldNode = path[pos];
        int n = path.length;
        int prev = path[(pos - 1 + n) % n];
        int next = path[(pos + 1) % n];

        int oldEdges = dist[prev][oldNode] + dist[oldNode][next];
        int newEdges = dist[prev][newNode] + dist[newNode][next];
        int costDiff = nodes[newNode].cost - nodes[oldNode].cost;

        return -oldEdges + newEdges + costDiff;
    }

    /**
     * Generate a random path of length k (i.e., select k distinct nodes).
     */
    private int[] generateRandomPath(int k, int n) {
        List<Integer> allNodes = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            allNodes.add(i);
        }
        Collections.shuffle(allNodes, random);
        List<Integer> chosen = allNodes.subList(0, k);
        Collections.shuffle(chosen, random);
        return chosen.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Simple random shuffle to "diversify" the path
     */
    private void randomShuffle(int[] path) {
        for (int i = path.length - 1; i > 0; i--) {
            int idx = random.nextInt(i + 1);
            int tmp = path[idx];
            path[idx] = path[i];
            path[i] = tmp;
        }
    }

    /**
     * REMOVAL OPERATORS (Destruction)
     */
    private interface RemovalOperator {
        List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance);
    }

    /**
     * 1) Random Removal: remove random positions from the path
     */
    private class RandomRemoval implements RemovalOperator {
        @Override
        public List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance) {
            List<Integer> removed = new ArrayList<>();
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < pathList.size(); i++) indices.add(i);
            Collections.shuffle(indices, random);
            for (int i = 0; i < removalCount; i++) {
                removed.add(pathList.get(indices.get(i)));
            }
            return removed;
        }
    }

    /**
     * 2) Distance-based Removal: remove the most "centrally distant" nodes
     *    (For example, pick a random pivot node, remove nodes that are far from pivot).
     */
    private class DistanceRemoval implements RemovalOperator {
        @Override
        public List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance) {
            if (pathList.isEmpty()) return new ArrayList<>();
            int pivotIndex = pathList.get(random.nextInt(pathList.size()));
            // Sort nodes by distance from pivot
            pathList.sort(Comparator.comparingInt(n -> -instance.distanceMatrix[pivotIndex][n]));
            return new ArrayList<>(pathList.subList(0, Math.min(removalCount, pathList.size())));
        }
    }

    /**
     * 3) Cost-based Removal: remove nodes with highest cost
     */
    private class CostRemoval implements RemovalOperator {
        @Override
        public List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance) {
            List<Integer> copy = new ArrayList<>(pathList);
            // Sort descending by node cost
            copy.sort(Comparator.comparingInt(n -> -instance.nodes[n].cost));
            return new ArrayList<>(copy.subList(0, Math.min(removalCount, copy.size())));
        }
    }

    /**
     * REPAIR OPERATOR: Weighted Regret. Reuse from earlier code but minimized for clarity.
     */
    private static class GreedyWeightedRegret {
        private final double w1 = 1.0;
        private final double w2 = 1.0;

        public Solution generateSolution(ProblemInstance instance, List<Integer> partialPath,
                                         Set<Integer> selected, int k) {
            Node[] nodes = instance.nodes;
            int n = nodes.length;
            int[][] dist = instance.distanceMatrix;
            List<Integer> path = new ArrayList<>(partialPath);

            while (path.size() < k) {
                int bestNode = -1;
                double bestWeighted = Double.NEGATIVE_INFINITY;
                int bestPos = -1;

                for (int node = 0; node < n; node++) {
                    if (selected.contains(node)) continue;
                    InsertionInfo info = findBestAndSecondBest(path, node, dist);
                    if (info == null) continue;

                    int regret = info.secondBest - info.best;
                    double weightedVal = w1 * regret - w2 * (info.best + nodes[node].cost);
                    if (weightedVal > bestWeighted) {
                        bestWeighted = weightedVal;
                        bestNode = node;
                        bestPos = info.position;
                    }
                }

                if (bestNode != -1 && bestPos != -1) {
                    path.add(bestPos, bestNode);
                    selected.add(bestNode);
                } else {
                    // no more possible insertions
                    break;
                }
            }

            int obj = computeObj(path.stream().mapToInt(i -> i).toArray(), instance);
            return new Solution(path.stream().mapToInt(i -> i).toArray(), obj);
        }

        private InsertionInfo findBestAndSecondBest(List<Integer> path, int node, int[][] dist) {
            if (path.isEmpty()) {
                // If path is empty, the cost increment is 0
                return new InsertionInfo(0, 0, 0);
            }
            int best = Integer.MAX_VALUE;
            int secondBest = Integer.MAX_VALUE;
            int bestPos = -1;
            int size = path.size();
            for (int i = 0; i < size; i++) {
                int current = path.get(i);
                int next = path.get((i + 1) % size);
                int inc = dist[current][node] + dist[node][next] - dist[current][next];

                if (inc < best) {
                    secondBest = best;
                    best = inc;
                    bestPos = i + 1;
                } else if (inc < secondBest) {
                    secondBest = inc;
                }
            }
            if (secondBest == Integer.MAX_VALUE) {
                secondBest = best;
            }
            return new InsertionInfo(bestPos, best, secondBest);
        }

        private int computeObj(int[] p, ProblemInstance instance) {
            int sumDist = 0;
            int[][] dist = instance.distanceMatrix;
            Node[] nodes = instance.nodes;
            for (int i = 0; i < p.length; i++) {
                sumDist += dist[p[i]][p[(i+1)%p.length]];
            }
            int sumCost = 0;
            for (int nodeIndex : p) {
                sumCost += nodes[nodeIndex].cost;
            }
            return sumDist + sumCost;
        }

        private static class InsertionInfo {
            int position;
            int best;
            int secondBest;
            public InsertionInfo(int position, int best, int secondBest) {
                this.position = position;
                this.best = best;
                this.secondBest = secondBest;
            }
        }
    }
}

/**
 * Utility class to compute statistics for a list of solutions.
 */
class StatisticsHelper {
    final int minObjective;
    final int maxObjective;
    final double avgObjective;
    final int[] bestPath;

    StatisticsHelper(int minObjective, int maxObjective, double avgObjective, int[] bestPath) {
        this.minObjective = minObjective;
        this.maxObjective = maxObjective;
        this.avgObjective = avgObjective;
        this.bestPath = bestPath;
    }

    static StatisticsHelper computeStatistics(List<Solution> solutions) {
        if (solutions == null || solutions.isEmpty()) {
            return new StatisticsHelper(0, 0, 0.0, new int[0]);
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
        return new StatisticsHelper(min, max, avg, bestPath);
    }
}

/**
 * Main class: processes each CSV instance, runs the single best ALNS method 20 times,
 * limited to ~870 ms each run, and outputs the results.
 */
class AdaptiveLargeNeighborhoodSearchMain {
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

        // Required: keep time at ~870 ms
        double maxTimeMs = 870.0;

        // We will do exactly 20 runs
        int runs = 20;

        for (File inputFile : inputFiles) {
            String fileName = inputFile.getName();
            String instanceName = fileName.substring(0, fileName.lastIndexOf('.'));
            String filePath = inputFile.getPath();

            System.out.println("Processing instance: " + instanceName);

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
                System.out.println("No valid nodes in CSV file '" + fileName + "'. Skipping.");
                continue;
            }
            // pick half the nodes
            int k = (int) Math.ceil(n / 2.0);
            System.out.println("Total nodes: " + n + ", Selecting k=" + k + " nodes.");

            // Create single best method with chosen parameters
            // For example:
            int maxNoImprovement = 200;   // # of iterations with no improvement before a random shuffle
            double removalFraction = 0.25; // fraction of nodes removed each iteration

            AdaptiveLargeNeighborhoodSearch alns = new AdaptiveLargeNeighborhoodSearch(maxNoImprovement, removalFraction);

            // We'll collect the solutions
            List<Solution> solutions = new ArrayList<>();

            for (int run = 0; run < runs; run++) {
                long startTime = System.nanoTime();
                // run ALNS
                Solution sol = alns.run(instance, k, maxTimeMs);
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;

                solutions.add(sol);
            }

            // Compute stats
            StatisticsHelper stats = StatisticsHelper.computeStatistics(solutions);

            System.out.println("\n--- Results for " + instanceName + " ---");
            System.out.println("Best Obj: " + stats.minObjective);
            System.out.println("Worst Obj: " + stats.maxObjective);
            System.out.printf("Avg Obj: %.2f%n", stats.avgObjective);
            System.out.println("Best Path: " + Arrays.toString(stats.bestPath));

            // Optionally save best path to CSV
            String outputDirPath = "outputs/" + instanceName;
            File outDir = new File(outputDirPath);
            if (!outDir.exists()) outDir.mkdirs();
            String bestPathFile = outputDirPath + "/ALNS_BestPath.csv";
            try {
                saveBestPathToCSV(stats.bestPath, bestPathFile);
                System.out.println("Best path saved to: " + bestPathFile + "\n");
            } catch (IOException e) {
                System.err.println("Error saving best path to CSV: " + e.getMessage());
            }
            System.out.println("***************************************\n");
        }
    }

    /**
     * Saves the best path to a CSV file, each node on a new line, repeating the first node at the end.
     */
    private static void saveBestPathToCSV(int[] bestPath, String fileName) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        for (int node : bestPath) {
            writer.write(Integer.toString(node));
            writer.newLine();
        }
        // optionally close the cycle
        if (bestPath.length > 0) {
            writer.write(Integer.toString(bestPath[0]));
            writer.newLine();
        }
        writer.close();
    }
}