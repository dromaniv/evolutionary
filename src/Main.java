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
 * A more advanced ALNS variant (v2) with:
 *  - Shaw Removal (relatedness-based) in addition to random/cost/distance
 *  - Multiple insertion operators (Weighted Regret, Cheapest Insertion)
 *  - Adaptive fraction of removal
 *  - 3-opt local search enhancements
 */
class AdvancedALNSv2 extends Heuristic {

    private final int maxNoImprovement;
    private final double minRemovalFrac;
    private final double maxRemovalFrac;

    // Removal operators
    private final List<RemovalOperator> removalOps;
    private final int[] removalScores;
    private final int[] removalAttempts;

    // Insertion operators
    private final List<InsertionOperator> insertionOps;
    private final int[] insertionScores;
    private final int[] insertionAttempts;

    /**
     * Constructor.
     *
     * @param maxNoImprovement how many iterations with no improvement before a random shuffle
     * @param minRemovalFrac   minimum fraction of nodes to remove
     * @param maxRemovalFrac   maximum fraction of nodes to remove
     */
    public AdvancedALNSv2(int maxNoImprovement, double minRemovalFrac, double maxRemovalFrac) {
        this.maxNoImprovement = maxNoImprovement;
        this.minRemovalFrac = minRemovalFrac;
        this.maxRemovalFrac = maxRemovalFrac;

        // Initialize removal operators
        removalOps = new ArrayList<>();
        removalOps.add(new RandomRemoval());
        removalOps.add(new DistanceRemoval());
        removalOps.add(new CostRemoval());
        removalOps.add(new ShawRemoval());  // new “relatedness” removal

        removalScores = new int[removalOps.size()];
        removalAttempts = new int[removalOps.size()];

        // Initialize insertion operators
        insertionOps = new ArrayList<>();
        insertionOps.add(new WeightedRegretInsertion());
        insertionOps.add(new CheapestInsertion());

        insertionScores = new int[insertionOps.size()];
        insertionAttempts = new int[insertionOps.size()];
    }

    /**
     * Executes ALNS within a given time limit (ms).
     */
    public Solution run(ProblemInstance instance, int k, double maxTimeMs) {
        long startTime = System.nanoTime();
        long maxDuration = (long)(maxTimeMs * 1e6);

        // 1) Create an initial random solution
        int[] initPath = generateRandomPath(k, instance.nodes.length);
        int initObj = computeObjective(initPath, instance);
        Solution bestSol = new Solution(initPath, initObj);

        // 2) Current solution
        int[] currPath = Arrays.copyOf(initPath, initPath.length);
        int currObj = initObj;

        int iterationSinceImprovement = 0;

        while ((System.nanoTime() - startTime) < maxDuration) {
            // (a) Decide how many nodes to remove this iteration (adaptive fraction)
            double alpha = random.nextDouble(); // random in [0,1]
            double removalFrac = minRemovalFrac + alpha * (maxRemovalFrac - minRemovalFrac);
            int removalCount = (int) Math.max(1, Math.floor(removalFrac * currPath.length));

            // (b) Pick removal & insertion operators via roulette selection
            RemovalOperator chosenRemovalOp = selectRemovalOperator();
            InsertionOperator chosenInsertionOp = selectInsertionOperator();

            // (c) Destroy
            int[] partial = removeNodes(currPath, instance, chosenRemovalOp, removalCount);

            // (d) Repair
            int[] repaired = chosenInsertionOp.insert(partial, instance, k);

            // (e) Local Search (with 2-opt & partial 3-opt)
            int[] improved = localSearch(repaired, instance);
            int improvedObj = computeObjective(improved, instance);

            // track usage of these operators
            int rIndex = removalOps.indexOf(chosenRemovalOp);
            int iIndex = insertionOps.indexOf(chosenInsertionOp);
            removalAttempts[rIndex]++;
            insertionAttempts[iIndex]++;

            // (f) Acceptance
            if (improvedObj < currObj) {
                // update current solution
                currPath = improved;
                currObj = improvedObj;
                // reward
                removalScores[rIndex] += 3;
                insertionScores[iIndex] += 3;
            }

            // check if we found a global best
            if (currObj < bestSol.objectiveValue) {
                bestSol = new Solution(currPath, currObj);
                iterationSinceImprovement = 0;
            } else {
                iterationSinceImprovement++;
            }

            // (g) If we exceed no improvement, random shuffle
            if (iterationSinceImprovement > maxNoImprovement) {
                randomShuffle(currPath);
                currObj = computeObjective(currPath, instance);
                iterationSinceImprovement = 0;
            }
        }

        return bestSol;
    }

    /**
     * Local Search with 2-opt and a partial 3-opt approach.
     */
    private int[] localSearch(int[] path, ProblemInstance instance) {
        int[] bestPath = Arrays.copyOf(path, path.length);
        int bestObj = computeObjective(bestPath, instance);
        boolean improved = true;

        while (improved) {
            improved = false;
            // 1) 2-opt
            for (int i = 0; i < bestPath.length; i++) {
                for (int j = i + 2; j < bestPath.length; j++) {
                    if (i == 0 && j == bestPath.length - 1) continue;
                    int delta = evaluate2OptDelta(bestPath, i, j, instance.distanceMatrix);
                    if (delta < 0) {
                        apply2Opt(bestPath, i, j);
                        bestObj += delta;
                        improved = true;
                    }
                }
            }

            // 2) Partial 3-opt: randomly pick a few triplets (i < j < k), attempt a standard 3-opt
            //   (We won't do a full 3-opt everywhere, just a few random picks to save time)
            for (int attempt = 0; attempt < 5; attempt++) {
                int i = random.nextInt(bestPath.length);
                int j = random.nextInt(bestPath.length);
                int k = random.nextInt(bestPath.length);
                if (i > j) {int t = i; i=j; j=t;}
                if (j > k) {int t=j; j=k; k=t;}
                if (i > j) {int t=i; i=j; j=t;} // recheck
                if (k - i < 3) continue; // skip too short segments

                int delta = evaluate3OptDelta(bestPath, i, j, k, instance.distanceMatrix);
                if (delta < 0) {
                    apply3Opt(bestPath, i, j, k);
                    bestObj += delta;
                    improved = true;
                }
            }
        }
        return bestPath;
    }

    /** Evaluate 2-opt delta for edges (i,i+1) and (j,j+1). */
    private int evaluate2OptDelta(int[] p, int i, int j, int[][] dist) {
        int n = p.length;
        int a = p[i];
        int b = p[(i+1) % n];
        int c = p[j];
        int d = p[(j+1) % n];
        return -dist[a][b] - dist[c][d] + dist[a][c] + dist[b][d];
    }

    /** Apply 2-opt: reverse segment p[i+1..j]. */
    private void apply2Opt(int[] p, int i, int j) {
        int n = p.length;
        int start = i + 1;
        int end = j;
        while (start < end) {
            int tmp = p[start];
            p[start] = p[end];
            p[end] = tmp;
            start++;
            end--;
        }
    }

    /**
     * Evaluate 3-opt delta for segments (i,i+1), (j,j+1), (k,k+1).
     * We only check a single 3-opt pattern for demonstration (there are 8 ways in full 3-opt).
     */
    private int evaluate3OptDelta(int[] p, int i, int j, int k, int[][] dist) {
        int n = p.length;
        int a = p[i], b = p[(i+1)%n], c = p[j], d = p[(j+1)%n], e = p[k], f = p[(k+1)%n];

        // We'll do a simple pattern: remove (a,b), (c,d), (e,f), reconnect as (a,d)-(c,f)-(e,b)
        // Typically you'd check multiple reconnection patterns for best improvement.
        int oldCost = dist[a][b] + dist[c][d] + dist[e][f];
        int newCost = dist[a][d] + dist[c][f] + dist[e][b];
        return newCost - oldCost;
    }

    /** Apply that single 3-opt pattern: segments: (a,b), (c,d), (e,f) => (a,d)-(c,f)-(e,b). */
    private void apply3Opt(int[] p, int i, int j, int k) {
        // For simplicity, we’ll do it with subarray reversals in a small-coded way:
        // A real 3-opt code would systematically reorder the segments p[i+1..j], p[j+1..k], etc.
        // We'll do a naive approach: rotate segments for the chosen pattern
        reverseSegment(p, j+1, k);  // (c,d) => reversed
        reverseSegment(p, i+1, j);  // (a,b) => reversed
        // disclaim: This is a simplified example of one 3-opt pattern
    }

    private void reverseSegment(int[] p, int start, int end) {
        while (start < end) {
            int tmp = p[start];
            p[start] = p[end];
            p[end] = tmp;
            start++;
            end--;
        }
    }

    /**
     * Removal & insertion operators
     */
    private RemovalOperator selectRemovalOperator() {
        double[] probabilities = computeOperatorProbabilities(removalScores, removalAttempts);
        double r = random.nextDouble();
        double cum = 0;
        for (int i = 0; i < probabilities.length; i++) {
            cum += probabilities[i];
            if (r <= cum) {
                return removalOps.get(i);
            }
        }
        return removalOps.get(probabilities.length - 1);
    }

    private InsertionOperator selectInsertionOperator() {
        double[] probabilities = computeOperatorProbabilities(insertionScores, insertionAttempts);
        double r = random.nextDouble();
        double cum = 0;
        for (int i = 0; i < probabilities.length; i++) {
            cum += probabilities[i];
            if (r <= cum) {
                return insertionOps.get(i);
            }
        }
        return insertionOps.get(probabilities.length - 1);
    }

    private double[] computeOperatorProbabilities(int[] scores, int[] attempts) {
        // Weighted by (score / attempts), fallback = 1 if attempts=0
        double[] prob = new double[scores.length];
        double sum = 0;
        for (int i = 0; i < scores.length; i++) {
            double ratio = (attempts[i] == 0) ? 1.0 : ((double)scores[i] / attempts[i]);
            prob[i] = ratio;
            sum += ratio;
        }
        // normalize
        for (int i = 0; i < prob.length; i++) {
            prob[i] /= sum;
        }
        return prob;
    }

    private int[] removeNodes(int[] path, ProblemInstance instance, RemovalOperator op, int removalCount) {
        List<Integer> pathList = Arrays.stream(path).boxed().collect(Collectors.toList());
        List<Integer> removed = op.selectNodesToRemove(pathList, removalCount, instance);
        pathList.removeAll(removed);
        return pathList.stream().mapToInt(Integer::intValue).toArray();
    }

    // Basic random path generator
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

    private void randomShuffle(int[] path) {
        for (int i = path.length - 1; i > 0; i--) {
            int idx = random.nextInt(i + 1);
            int tmp = path[idx];
            path[idx] = path[i];
            path[i] = tmp;
        }
    }

    /**
     * Removal Operators
     */
    private interface RemovalOperator {
        List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance);
    }

    // 1) Random Removal
    private class RandomRemoval implements RemovalOperator {
        @Override
        public List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance) {
            List<Integer> removed = new ArrayList<>();
            List<Integer> idxs = new ArrayList<>();
            for (int i = 0; i < pathList.size(); i++) idxs.add(i);
            Collections.shuffle(idxs, random);
            for (int i = 0; i < removalCount; i++) {
                removed.add(pathList.get(idxs.get(i)));
            }
            return removed;
        }
    }

    // 2) Distance-based removal
    private class DistanceRemoval implements RemovalOperator {
        @Override
        public List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance) {
            if (pathList.isEmpty()) return new ArrayList<>();
            int pivot = pathList.get(random.nextInt(pathList.size()));
            pathList.sort(Comparator.comparingInt(n -> -instance.distanceMatrix[pivot][n])); // descending distance
            return new ArrayList<>(pathList.subList(0, Math.min(removalCount, pathList.size())));
        }
    }

    // 3) Cost-based removal
    private class CostRemoval implements RemovalOperator {
        @Override
        public List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance) {
            List<Integer> copy = new ArrayList<>(pathList);
            copy.sort(Comparator.comparingInt(n -> -instance.nodes[n].cost)); // highest cost first
            return new ArrayList<>(copy.subList(0, Math.min(removalCount, copy.size())));
        }
    }

    // 4) Shaw (Relatedness) Removal: remove nodes that are pairwise “close” in coordinate or cost
    private class ShawRemoval implements RemovalOperator {
        @Override
        public List<Integer> selectNodesToRemove(List<Integer> pathList, int removalCount, ProblemInstance instance) {
            if (pathList.isEmpty()) return new ArrayList<>();
            // pick random “seed” node
            int seedIndex = pathList.get(random.nextInt(pathList.size()));
            // compute relatedness = combination of distance and cost difference
            // higher => more related => more likely to remove
            // for simplicity: relatedness = distance(seedIndex, n) + |cost(seedIndex) - cost(n)|
            Map<Integer, Double> relatednessMap = new HashMap<>();
            Node seed = instance.nodes[seedIndex];
            for (int nd : pathList) {
                Node ndN = instance.nodes[nd];
                int dist = instance.distanceMatrix[seedIndex][nd];
                double costDiff = Math.abs(seed.cost - ndN.cost);
                double relatedness = dist + costDiff;
                relatednessMap.put(nd, relatedness);
            }
            // sort ascending by relatedness so the "most related" are at the front
            // we'll remove those that are "closest" to the seed
            List<Integer> sorted = new ArrayList<>(pathList);
            sorted.sort(Comparator.comparingDouble(relatednessMap::get));
            return new ArrayList<>(sorted.subList(0, Math.min(removalCount, sorted.size())));
        }
    }

    /**
     * Insertion Operators
     */
    private interface InsertionOperator {
        int[] insert(int[] partialPath, ProblemInstance instance, int k);
    }

    // Weighted Regret from before
    private class WeightedRegretInsertion implements InsertionOperator {
        private final double w1 = 1.0;
        private final double w2 = 1.0;

        @Override
        public int[] insert(int[] partialPath, ProblemInstance instance, int k) {
            List<Integer> pathList = Arrays.stream(partialPath).boxed().collect(Collectors.toList());
            Set<Integer> used = new HashSet<>(pathList);
            int n = instance.nodes.length;
            while (pathList.size() < k) {
                int bestNode = -1;
                double bestWeighted = Double.NEGATIVE_INFINITY;
                int bestPos = -1;

                for (int nd = 0; nd < n; nd++) {
                    if (used.contains(nd)) continue;
                    InsertionInfo info = findBestAndSecondBest(pathList, nd, instance);
                    if (info == null) continue;

                    int regret = info.secondBest - info.best;
                    double weightedVal = w1 * regret - w2*(info.best + instance.nodes[nd].cost);
                    if (weightedVal > bestWeighted) {
                        bestWeighted = weightedVal;
                        bestNode = nd;
                        bestPos = info.position;
                    }
                }
                if (bestNode == -1) break;
                pathList.add(bestPos, bestNode);
                used.add(bestNode);
            }
            return pathList.stream().mapToInt(i->i).toArray();
        }

        private InsertionInfo findBestAndSecondBest(List<Integer> path, int node, ProblemInstance instance) {
            if (path.isEmpty()) {
                return new InsertionInfo(0, 0, 0);
            }
            int[][] dist = instance.distanceMatrix;
            int best = Integer.MAX_VALUE;
            int secondBest = Integer.MAX_VALUE;
            int bestPos = -1;
            int sz = path.size();
            for (int i = 0; i < sz; i++) {
                int curr = path.get(i);
                int nxt = path.get((i+1) % sz);
                int inc = dist[curr][node] + dist[node][nxt] - dist[curr][nxt];
                if (inc < best) {
                    secondBest = best;
                    best = inc;
                    bestPos = i+1;
                } else if (inc < secondBest) {
                    secondBest = inc;
                }
            }
            if (secondBest == Integer.MAX_VALUE) {
                secondBest = best;
            }
            return new InsertionInfo(bestPos, best, secondBest);
        }
        private class InsertionInfo {
            int position, best, secondBest;
            InsertionInfo(int pos, int b, int sb) {
                position = pos; best = b; secondBest = sb;
            }
        }
    }

    // Cheapest Insertion
    private class CheapestInsertion implements InsertionOperator {
        @Override
        public int[] insert(int[] partialPath, ProblemInstance instance, int k) {
            List<Integer> pathList = Arrays.stream(partialPath).boxed().collect(Collectors.toList());
            Set<Integer> used = new HashSet<>(pathList);
            int n = instance.nodes.length;
            int[][] dist = instance.distanceMatrix;
            Node[] nodes = instance.nodes;

            while (pathList.size() < k) {
                int bestNode = -1;
                int bestCost = Integer.MAX_VALUE;
                int bestPos = -1;

                for (int nd = 0; nd < n; nd++) {
                    if (used.contains(nd)) continue;
                    // find insertion cost
                    InsertionPos ip = findCheapestPos(pathList, nd, dist);
                    if (ip.cost + nodes[nd].cost < bestCost) {
                        bestCost = ip.cost + nodes[nd].cost;
                        bestNode = nd;
                        bestPos = ip.pos;
                    }
                }

                if (bestNode == -1) break;
                pathList.add(bestPos, bestNode);
                used.add(bestNode);
            }
            return pathList.stream().mapToInt(i->i).toArray();
        }

        private InsertionPos findCheapestPos(List<Integer> path, int node, int[][] dist) {
            if (path.isEmpty()) return new InsertionPos(0, 0);
            int best = Integer.MAX_VALUE;
            int pos = -1;
            int sz = path.size();
            for (int i = 0; i < sz; i++) {
                int curr = path.get(i);
                int nxt = path.get((i+1)%sz);
                int inc = dist[curr][node] + dist[node][nxt] - dist[curr][nxt];
                if (inc < best) {
                    best = inc;
                    pos = i+1;
                }
            }
            return new InsertionPos(pos, best);
        }

        private class InsertionPos {
            int pos, cost;
            InsertionPos(int p, int c) {
                pos = p; cost = c;
            }
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
 * Main: processes each CSV instance and runs the single AdvancedALNSv2 method 20 times,
 * each with a ~870 ms time limit, logging results.
 */
class Main {
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

        // We keep the ~870 ms limit
        double maxTimeMs = 870.0;
        // We do exactly 20 runs
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

            // as before, select half the nodes
            int k = (int) Math.ceil(n / 2.0);
            System.out.println("Total nodes: " + n + ", Selecting k=" + k + " nodes.");

            // Build the single best ALNS with advanced features
            // Example parameters:
            int maxNoImprovement = 200;   // # iterations with no improvement
            double minRemovalFrac = 0.15;
            double maxRemovalFrac = 0.35;

            AdvancedALNSv2 alns = new AdvancedALNSv2(maxNoImprovement, minRemovalFrac, maxRemovalFrac);

            // Collect solutions
            List<Solution> solutions = new ArrayList<>();

            for (int run = 0; run < runs; run++) {
                long startTime = System.nanoTime();
                // run advanced ALNS
                Solution sol = alns.run(instance, k, maxTimeMs);
                long endTime = System.nanoTime();
                double durationMs = (endTime - startTime) / 1e6;

                solutions.add(sol);
            }

            // Compute stats
            Statistics stats = Statistics.computeStatistics(solutions);

            System.out.println("\n--- Results for " + instanceName + " ---");
            System.out.println("Best Obj: " + stats.minObjective);
            System.out.println("Worst Obj: " + stats.maxObjective);
            System.out.printf("Avg Obj: %.2f%n", stats.avgObjective);
            System.out.println("Best Path: " + Arrays.toString(stats.bestPath));

            // Save the best path
            String outputDirPath = "outputs/" + instanceName;
            File outDir = new File(outputDirPath);
            if (!outDir.exists()) outDir.mkdirs();
            String bestPathFile = outputDirPath + "/AdvancedALNSv2_BestPath.csv";
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
     * Saves the best path to a CSV file, each node on a new line, optionally repeating the first node at the end.
     */
    private static void saveBestPathToCSV(int[] bestPath, String fileName) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        for (int node : bestPath) {
            writer.write(Integer.toString(node));
            writer.newLine();
        }
        if (bestPath.length > 0) {
            writer.write(Integer.toString(bestPath[0]));
            writer.newLine();
        }
        writer.close();
    }
}