package neural_network.learning;

import neural_network.components.Network;
import neural_network.util.Header;

import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;

/** Class to test a neural network
 *
 */
public class Tester extends Learner {

    /** General constructor - matches {@code Learner}.
     *
     * @param network The neural network to train.
     * @param data All the data to be passed to the {@code network}.
     * @param batchSize The number of datapoints per batch for an epoch.
     * @param weighted If {@code true} then we use the {@code WeightedPartitioner},
     *                 else we use the {@code Partitioner}.
     * @param numBins If {@code weighted} is {@code true} and the {@code network} is
     *                a regressor, then we need to specify the number of bins for the
     *                {@code WeightedPartitioner}. Otherwise, this parameter is ignored.
     */
    public Tester(Network network, NavigableMap<Header, List<String>> data, int batchSize,
                     boolean weighted, int numBins) {
        super(network, data, batchSize, weighted, numBins);
    }

    /** Default constructor for non-weighted partitions - matches {@code Learner}.
     *
     * @param network The neural network to train.
     * @param data All the data to be passed to the {@code network}.
     * @param batchSize The number of datapoints per batch for an epoch.
     */
    public Tester(Network network, NavigableMap<Header, List<String>> data, int batchSize) {
        super(network, data, batchSize);
    }

    /** Performs testing of the network.
     *
     */
    @Override
    public void run() {
        double totalLoss = 0.0;
        List<List<Integer>> batchPartition = getPartitioner().call();
        int itsPerEpoch = (int) Math
                .ceil((double) getNumDatapoints() / getBatchSize());
        for (int iteration = 0; iteration < itsPerEpoch; iteration ++) {
            List<Integer> batchIds = batchPartition.get(iteration);

            // Do forward pass and back propagation for this specific batch
            totalLoss += forwardPassOneBatch(batchIds);
        }
        double loss = Math.round(10000 * totalLoss / getNumDatapoints()) / 10000.0;
        System.out.printf("Testing loss: %.4f%n", loss);

        if (! isRegressor()) {
            updateCategoricalDataframe();
        }
    }

    /** Creates scatter plot from the data and their predicted values.
     *
     * @param title An optional title to append to the plot.
     * @throws IOException If an IO error occurs.
     */
    public void generateScatter(String title) throws IOException {
        super.generateScatter("testing", title);
    }

    /** Creates scatter plot comparing predicted to actual values (for
     * regression problems only).
     *
     * @param title An optional title to append to the plot.
     * @throws IOException If an IO error occurs.
     * @throws RuntimeException If this method is called with a categorical network,
     *                          instead user should call {@code Tester.generateConfusion()}.
     */
    public void comparisonScatter(String title) throws IOException {
        super.comparisonScatter("testing", title);
    }

    /** Creates a confusion matrix and dice scores from the results.
     *
     */
    public void generateConfusion() {
        List<String> categoryNames = getCategoryNames();
        int numCategories = categoryNames.size();
        // Below is a matrix of 0s with size numCategories
        List<List<Integer>> contingencyTable = new ArrayList<>();
        List<String> y = getCategoricalDf().get(Header.Y);
        List<String> yHat = getCategoricalDf().get(Header.Y_HAT);
        // All y and y hat pairs
        List<List<String>> pairs = Stream
                .iterate(0, i -> i < getNumDatapoints(), i -> i + 1)
                .map(i -> List.of(y.get(i), yHat.get(i)))
                .toList();
        // Setting up the frequencies of pairs in the contingency table
        for (int i = 0; i < numCategories; i ++) {
            String yCategory = categoryNames.get(i);
            contingencyTable.add(new ArrayList<>());
            for (int j = 0; j < numCategories; j ++) {
                String yHatCategory = categoryNames.get(j);
                List<String> pair = List.of(yCategory, yHatCategory);
                contingencyTable.get(i).add(Collections.frequency(pairs, pair));
            }
        }
        printConfusion(contingencyTable);
        printDiceScores(contingencyTable);
    }

    /** Prints out the confusion matrix from a contingency table.
     *
     * @param contingencyTable Pairwise frequency data for y and y hat values.
     */
    void printConfusion(List<List<Integer>> contingencyTable) {
        System.out.println("Confusion matrix");
        System.out.println("-----------------------------------------------");
        List<String> categoryNames = getCategoryNames();
        int numCategories = categoryNames.size();
        int maxLength = Collections.max(categoryNames.stream()
                .map(String::length).toList());
        String strCategoryNames = String.join("  ", categoryNames);
        System.out.printf("%-5s%-8s%-10s%n", "", "", "yHat");
        System.out.printf("%-5s%-8s%-10s%n", "", "", strCategoryNames);
        int halfWay = numCategories / 2;
        for (int i = 0; i < numCategories; i ++) {
            String yTitle = (i == halfWay) ? "y" : "";
            String strRow = String.join(" ".repeat(maxLength), contingencyTable.get(i).stream()
                    .map(Object::toString).toList());
            System.out.printf("%-5s%-8s%-10s%n", yTitle, categoryNames.get(i), strRow);
        }
        System.out.println("-----------------------------------------------");
    }

    /** Prints out the dice scores from a contingency table.
     *
     * @param contingencyTable Pairwise frequency data for y and y hat values.
     */
    void printDiceScores(List<List<Integer>> contingencyTable) {
        NavigableMap<String, Double> diceScores = new TreeMap<>();
        int n = contingencyTable.size();
        for (int i = 0; i < n; i ++) {
            int truePositive = contingencyTable.get(i).get(i);
            int sumRow = contingencyTable.get(i).stream().mapToInt(Integer::intValue).sum();
            int finalI = i;
            int sumColumn = Stream
                    .iterate(0, j -> j < n, j -> j + 1)
                    .mapToInt(j -> contingencyTable.get(j).get(finalI))
                    .sum();
            double diceScore = Math.round((2.0 * truePositive * 10000) / (sumRow + sumColumn))
                    / 10000.0;
            diceScores.put(getCategoryNames().get(i), diceScore);
        }
        System.out.println("Dice scores: " + diceScores);
        double meanDiceScore = diceScores.values().stream()
                .mapToDouble(Double::doubleValue).sum() / n;
        double roundedScore = Math.round(meanDiceScore * 10000) / 10000.0;
        System.out.println("Mean dice score: " + roundedScore);
    }
}
