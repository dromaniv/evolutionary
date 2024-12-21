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
                    iterator.remove();
                    continue;
                }

                int delta = move.computeDelta(currentPath, distanceMatrix, nodes);
                if (delta < 0) {
                    move.apply(currentPath, inPathSet);
                    improvement = true;
                    iterator.remove();
                    break;
                }
            }

            if (improvement) {
                continue;
            }

            // If no improvement from previous moves, explore the full neighborhood
            int bestDelta = 0;
            Move bestMove = null;

            int nPath = currentPath.length;
            // Intra-route moves (2-opt)
            for (int i = 0; i < nPath; i++) {
                int a = currentPath[i];
                int b = currentPath[(i + 1) % nPath];
                for (int j = i + 2; j < nPath; j++) {
                    if (i == 0 && j == nPath - 1) {
                        continue; // do not reverse the entire path
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
                        continue;
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
                bestMove.apply(currentPath, inPathSet);
                improvement = true;
                improvingMoves.clear();
                improvingMoves.add(new Move(bestMove));
            }
        }

        int objective = computeObjective(currentPath, instance);
        return new Solution(currentPath, objective);
    }

    public Solution generateSolution(ProblemInstance instance, int k, int maxIterations) {
        Node[] nodes = instance.nodes;
        int n = nodes.length;
        int[] allNodes = new int[n];
        for (int i = 0; i < n; i++) {
            allNodes[i] = i;
        }

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

    /**
     * Inner class representing a move in local search.
     */
    private static class Move {
        int index1;
        int index2;
        MoveType type;

        int nodeA, nodeB, nodeC, nodeD;
        int nodeU;

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

        void setInterRouteMove(int index1, int nodeV, int nodeU) {
            this.index1 = index1;
            this.nodeU = nodeU;
            this.type = MoveType.INTER_ROUTE;
            // nodeV is the node to be inserted
            this.nodeA = nodeV; // Reusing nodeA to store nodeV
        }

        boolean isValid(int[] path, boolean[] inPathSet) {
            if (type == MoveType.INTRA_ROUTE) {
                int nPath = path.length;
                return path[index1] == nodeA && path[(index1 + 1) % nPath] == nodeB &&
                        path[index2] == nodeC && path[(index2 + 1) % nPath] == nodeD;
            } else if (type == MoveType.INTER_ROUTE) {
                return path[index1] == nodeU && !inPathSet[nodeA];
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
                int v = nodeA;
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
                int v = nodeA;

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
 */
class GreedyWeightedRegret extends Heuristic {
    private final double w1;
    private final double w2;

    public GreedyWeightedRegret() {
        this.w1 = 1;
        this.w2 = 1;
    }

    protected InsertionInfo findBestAndSecondBestInsertion(List<Integer> path, int nodeToInsert, int[][] distanceMatrix) {
        if (path.isEmpty()) {
            return new InsertionInfo(0, 0, 0);
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

        if (secondBestIncrease == Integer.MAX_VALUE) {
            secondBestIncrease = bestIncrease;
        }

        return new InsertionInfo(bestPos, bestIncrease, secondBestIncrease);
    }

    public Solution generateSolution(ProblemInstance instance, List<Integer> partialPath, Set<Integer> selected, int k) {
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
                if (insertionInfo == null)
                    continue;

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
                // No more nodes to add
                break;
            }
        }

        int objective = computeObjective(path.stream().mapToInt(Integer::intValue).toArray(), instance);
        return new Solution(path.stream().mapToInt(Integer::intValue).toArray(), objective);
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
 * Implements a Hybrid Evolutionary Algorithm (HEA).
 */
class HybridEvolutionaryAlgorithm extends Heuristic {

    private final ProblemInstance instance;
    private final int k;
    private final double maxTimeMs;
    private final int populationSize = 20;  // Elite population of 20
    private final boolean applyLocalSearchToOffspring;
    private final SteepestLocalSearchWithMoveEvaluation localSearch;
    private final GreedyWeightedRegret repairHeuristic;
    private final Random rand = new Random();

    public int numIterations; // number of iterations performed

    /**
     * Constructor.
     *
     * @param instance                    Problem instance
     * @param k                           Number of nodes in the solution
     * @param maxTimeMs                   Maximum time in milliseconds
     * @param applyLocalSearchToOffspring Whether to apply local search after each recombination
     */
    public HybridEvolutionaryAlgorithm(ProblemInstance instance, int k, double maxTimeMs,
                                       boolean applyLocalSearchToOffspring) {
        this.instance = instance;
        this.k = k;
        this.maxTimeMs = maxTimeMs;
        this.applyLocalSearchToOffspring = applyLocalSearchToOffspring;
        this.localSearch = new SteepestLocalSearchWithMoveEvaluation();
        this.repairHeuristic = new GreedyWeightedRegret();
    }

    /**
     * Runs the hybrid evolutionary algorithm.
     *
     * @return The best solution found.
     */
    public Solution run() {
        long startTime = System.nanoTime();
        long maxDuration = (long) (maxTimeMs * 1e6); // ms to ns

        // 1. Generate initial population (apply local search for each individual).
        List<Solution> population = initializePopulation();

        // 2. Keep track of the best solution found so far.
        Solution bestSolution = findBest(population);

        numIterations = 0;
        while ((System.nanoTime() - startTime) < maxDuration) {
            numIterations++;

            // 3. Select parents with uniform probability from the population
            Solution parent1 = population.get(rand.nextInt(population.size()));
            Solution parent2 = population.get(rand.nextInt(population.size()));

            // 4. Apply recombination to produce a single offspring
            Solution offspring = recombine(parent1, parent2);

            // 5. Optionally apply local search on the offspring
            if (applyLocalSearchToOffspring) {
                offspring = localSearch.generateSolutionFromPath(instance, offspring.path, Integer.MAX_VALUE);
            }

            // 6. Insert offspring into population if it is not a duplicate
            if (!isDuplicate(population, offspring)) {
                replaceWorstSolution(population, offspring);
                // Update best if offspring is better
                if (offspring.objectiveValue < bestSolution.objectiveValue) {
                    bestSolution = offspring;
                }
            }

            // Termination condition based on time, so we continue until time is up.
        }

        return bestSolution;
    }

    /**
     * Generates the initial population by randomly creating solutions
     * and applying local search to each.
     */
    private List<Solution> initializePopulation() {
        List<Solution> population = new ArrayList<>(populationSize);
        int n = instance.nodes.length;
        int[] allNodes = new int[n];
        for (int i = 0; i < n; i++) {
            allNodes[i] = i;
        }

        // We generate 'populationSize' solutions
        while (population.size() < populationSize) {
            shuffleArray(allNodes);
            // pick the first k nodes
            int[] partialPath = Arrays.copyOfRange(allNodes, 0, k);
            shuffleArray(partialPath);

            // local search on the random path
            Solution sol = localSearch.generateSolutionFromPath(instance, partialPath, Integer.MAX_VALUE);

            if (!isDuplicate(population, sol)) {
                population.add(sol);
            }
        }
        return population;
    }

    /**
     * Two recombination operators:
     *   Operator 1: Common edges.
     *   Operator 2: Remove from parent1 the edges not in parent2, then repair.
     * We choose one of these operators randomly (50-50 probability).
     *
     * @param parent1 A solution
     * @param parent2 Another solution
     * @return Offspring solution
     */
    private Solution recombine(Solution parent1, Solution parent2) {
        // Choose randomly between operator 1 and operator 2
        if (rand.nextBoolean()) {
            return recombineOperator1(parent1, parent2);
        } else {
            return recombineOperator2(parent1, parent2);
        }
    }

    /**
     * Recombination Operator 1:
     *  - The offspring inherits all edges that appear in both parents.
     *  - The rest of the nodes are added randomly until we have k nodes.
     */
    private Solution recombineOperator1(Solution parent1, Solution parent2) {
        // Convert paths to sets of edges for quick membership check
        Set<Edge> edgesParent1 = getEdgeSet(parent1.path);
        Set<Edge> edgesParent2 = getEdgeSet(parent2.path);

        // The common edges are the intersection
        Set<Edge> commonEdges = new HashSet<>(edgesParent1);
        commonEdges.retainAll(edgesParent2);

        // Collect unique nodes from common edges
        Set<Integer> nodesInEdges = new HashSet<>();
        for (Edge e : commonEdges) {
            nodesInEdges.add(e.u);
            nodesInEdges.add(e.v);
        }

        List<Integer> nodeList = new ArrayList<>(nodesInEdges);
        Collections.shuffle(nodeList, rand);

        // If we already have more than k, we'll trim randomly
        while (nodeList.size() > k) {
            nodeList.remove(nodeList.size() - 1);
        }

        // If we have fewer than k, add random nodes not yet in the set
        int n = instance.nodes.length;
        for (int i = 0; i < n && nodeList.size() < k; i++) {
            if (!nodeList.contains(i)) {
                nodeList.add(i);
            }
        }

        // Now we have exactly k nodes. We must form a path or cycle from them.
        // We can simply shuffle them or perform a small greedy for edges.
        // Here, let's just shuffle them.
        Collections.shuffle(nodeList, rand);

        int[] childArr = nodeList.stream().mapToInt(i -> i).toArray();
        int objValue = computeObjective(childArr, instance);
        return new Solution(childArr, objValue);
    }

    /**
     * Recombination Operator 2:
     *  - Pick parent1 as the base.
     *  - Remove from it all nodes (edges) that are not in parent2.
     *  - Repair the solution using the same approach as in LNS (Greedy Weighted Regret).
     */
    private Solution recombineOperator2(Solution parent1, Solution parent2) {
        // Convert parent2 path into a set for quick membership check
        Set<Integer> parent2Set = Arrays.stream(parent2.path).boxed().collect(Collectors.toSet());

        // Copy parent1's path
        List<Integer> childPath = Arrays.stream(parent1.path).boxed().collect(Collectors.toList());

        // Remove nodes in childPath not in parent2
        childPath.removeIf(node -> !parent2Set.contains(node));

        // Now we have a partial solution
        // Use Greedy Weighted Regret to repair up to k nodes
        Set<Integer> selected = new HashSet<>(childPath);
        // Reuse the same 'repairHeuristic' approach from LNS
        Solution repaired = repairHeuristic.generateSolution(instance, childPath, selected, k);
        int[] repairedPath = repaired.path;

        return new Solution(repairedPath, repaired.objectiveValue);
    }

    /**
     * Replaces the worst solution in the population (steady-state) with the new offspring.
     * Alternatively, you could replace a random or near-worst solution.
     */
    private void replaceWorstSolution(List<Solution> population, Solution offspring) {
        int worstIndex = 0;
        int worstObj = population.get(0).objectiveValue;
        for (int i = 1; i < population.size(); i++) {
            if (population.get(i).objectiveValue > worstObj) {
                worstObj = population.get(i).objectiveValue;
                worstIndex = i;
            }
        }
        population.set(worstIndex, offspring);
    }

    /**
     * Checks whether 'offspring' is a duplicate of any solution in 'population'.
     * We can compare by objective value or by checking identical path.
     * Here, to be safe, we check both.
     */
    private boolean isDuplicate(List<Solution> population, Solution offspring) {
        for (Solution sol : population) {
            // Check objective equality
            if (sol.objectiveValue == offspring.objectiveValue) {
                // Also check if path is the same set of nodes (order not necessarily the same)
                if (samePath(sol.path, offspring.path)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Checks if two solutions contain the same path (same set of nodes).
     * You can refine this check (e.g., sort, or check rotation invariances).
     */
    private boolean samePath(int[] path1, int[] path2) {
        if (path1.length != path2.length)
            return false;
        Set<Integer> s1 = Arrays.stream(path1).boxed().collect(Collectors.toSet());
        Set<Integer> s2 = Arrays.stream(path2).boxed().collect(Collectors.toSet());
        return s1.equals(s2);
    }

    /**
     * Finds the solution with minimum objective in the list.
     */
    private Solution findBest(List<Solution> solutions) {
        Solution best = solutions.get(0);
        for (Solution s : solutions) {
            if (s.objectiveValue < best.objectiveValue) {
                best = s;
            }
        }
        return best;
    }

    /**
     * Converts a path into a set of edges (u,v) pairs.
     */
    private Set<Edge> getEdgeSet(int[] path) {
        Set<Edge> edges = new HashSet<>();
        int n = path.length;
        for (int i = 0; i < n; i++) {
            int u = path[i];
            int v = path[(i + 1) % n];
            // store undirected edge (ensure smaller index first if you want to treat it as undirected)
            if (u < v)
                edges.add(new Edge(u, v));
            else
                edges.add(new Edge(v, u));
        }
        return edges;
    }

    /**
     * Shuffle an array in place.
     */
    private void shuffleArray(int[] array) {
        int index, temp;
        for (int i = array.length - 1; i > 0; i--) {
            index = rand.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    /**
     * Simple Edge class for storing edges in an undirected manner.
     */
    private static class Edge {
        int u;
        int v;

        Edge(int u, int v) {
            this.u = u;
            this.v = v;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (!(obj instanceof Edge))
                return false;
            Edge other = (Edge) obj;
            return (this.u == other.u && this.v == other.v);
        }

        @Override
        public int hashCode() {
            // typical way to combine two ints for a hash
            return 31 * u + v;
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
 * Implements a Hybrid Evolutionary Algorithm (HEA).
 */
class HybridEvolutionaryAlgorithmMain {
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

        // Use average running time from previous problem (~870 ms)
        double maxTimeMs = 870.0;

        // Process each input file
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
                System.out.println("No valid nodes found in the CSV file '" + fileName + "'. Skipping.");
                continue;
            }
            int k = (int) Math.ceil(n / 2.0);
            System.out.println("Total nodes: " + n + ", Selecting k=" + k + " nodes.\n");

            String outputInstanceDirPath = "outputs/" + instanceName;
            File outputInstanceDir = new File(outputInstanceDirPath);
            if (!outputInstanceDir.exists()) {
                boolean created = outputInstanceDir.mkdirs();
                if (!created) {
                    System.err.println("Failed to create output directory for instance '" + instanceName + "'. Skipping.");
                    continue;
                }
            }

            // Running Hybrid Evolutionary Algorithm (HEA) with local search after recombination
            System.out.println("Running Hybrid Evolutionary Algorithm (HEA) with local search after recombination...");
            List<Solution> heaWithLSsolutions = new ArrayList<>();
            List<Integer> heaWithLSIterations = new ArrayList<>();
            double totalHeaWithLsTime = 0.0;

            for (int run = 0; run < 20; run++) {
                HybridEvolutionaryAlgorithm hea = new HybridEvolutionaryAlgorithm(instance, k, maxTimeMs, true);
                long startTime = System.nanoTime();
                Solution sol = hea.run();
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;
                totalHeaWithLsTime += durationMs;

                heaWithLSsolutions.add(sol);
                heaWithLSIterations.add(hea.numIterations);
            }

            double avgHeaWithLsTime = totalHeaWithLsTime / 20;
            double avgHeaWithLsIterations = heaWithLSIterations.stream().mapToDouble(a -> a).average().orElse(0.0);
            StatisticsHelper heaWithLSStats = StatisticsHelper.computeStatistics(heaWithLSsolutions);

            // Running Hybrid Evolutionary Algorithm (HEA) without local search after recombination
            System.out.println("Running Hybrid Evolutionary Algorithm (HEA) without local search after recombination...");
            List<Solution> heaNoLSsolutions = new ArrayList<>();
            List<Integer> heaNoLSIterations = new ArrayList<>();
            double totalHeaNoLsTime = 0.0;

            for (int run = 0; run < 20; run++) {
                HybridEvolutionaryAlgorithm hea = new HybridEvolutionaryAlgorithm(instance, k, maxTimeMs, false);
                long startTime = System.nanoTime();
                Solution sol = hea.run();
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;
                totalHeaNoLsTime += durationMs;

                heaNoLSsolutions.add(sol);
                heaNoLSIterations.add(hea.numIterations);
            }

            double avgHeaNoLsTime = totalHeaNoLsTime / 20;
            double avgHeaNoLsIterations = heaNoLSIterations.stream().mapToDouble(a -> a).average().orElse(0.0);
            StatisticsHelper heaNoLSStats = StatisticsHelper.computeStatistics(heaNoLSsolutions);

            // Output results
            System.out.println("\n--- Computational Results for Instance: " + instanceName + " ---\n");

            // HEA with Local Search
            System.out.println("Method: HEA with local search after recombination");
            System.out.println("Min Objective: " + heaWithLSStats.minObjective);
            System.out.println("Max Objective: " + heaWithLSStats.maxObjective);
            System.out.printf("Average Objective: %.2f%n", heaWithLSStats.avgObjective);
            System.out.printf("Average Time per run: %.2f ms%n", avgHeaWithLsTime);
            System.out.printf("Average Number of Iterations: %.2f%n", avgHeaWithLsIterations);
            System.out.println("Best Solution Path: " + Arrays.toString(heaWithLSStats.bestPath) + "\n");

            String heaWithLSFileName = outputInstanceDirPath + "/HEA_with_LS.csv";
            try {
                saveBestPathToCSV(heaWithLSStats.bestPath, heaWithLSFileName);
                System.out.println("Best path for HEA with LS saved to " + heaWithLSFileName + "\n");
            } catch (IOException e) {
                System.err.println("Error writing best path to CSV for HEA with LS: " + e.getMessage());
            }

            // HEA without Local Search
            System.out.println("Method: HEA without local search after recombination");
            System.out.println("Min Objective: " + heaNoLSStats.minObjective);
            System.out.println("Max Objective: " + heaNoLSStats.maxObjective);
            System.out.printf("Average Objective: %.2f%n", heaNoLSStats.avgObjective);
            System.out.printf("Average Time per run: %.2f ms%n", avgHeaNoLsTime);
            System.out.printf("Average Number of Iterations: %.2f%n", avgHeaNoLsIterations);
            System.out.println("Best Solution Path: " + Arrays.toString(heaNoLSStats.bestPath) + "\n");

            String heaNoLSFileName = outputInstanceDirPath + "/HEA_no_LS.csv";
            try {
                saveBestPathToCSV(heaNoLSStats.bestPath, heaNoLSFileName);
                System.out.println("Best path for HEA without LS saved to " + heaNoLSFileName + "\n");
            } catch (IOException e) {
                System.err.println("Error writing best path to CSV for HEA without LS: " + e.getMessage());
            }

            System.out.println("Finished processing instance: " + instanceName + "\n");
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