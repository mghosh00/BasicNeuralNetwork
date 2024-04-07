package neural_network.learning;

import neural_network.components.Network;
import neural_network.util.Header;

import java.io.IOException;
import java.util.List;
import java.util.NavigableMap;

/** Class to validate a neural network.
 *
 */
public class Validator extends Learner {

    private int epoch = 0;

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
    public Validator(Network network, NavigableMap<Header, List<String>> data, int batchSize,
                     boolean weighted, int numBins) {
        super(network, data, batchSize, weighted, numBins);
    }

    /** Default constructor for non-weighted partitions - matches {@code Learner}.
     *
     * @param network The neural network to train.
     * @param data All the data to be passed to the {@code network}.
     * @param batchSize The number of datapoints per batch for an epoch.
     */
    public Validator(Network network, NavigableMap<Header, List<String>> data, int batchSize) {
        super(network, data, batchSize);
    }

    /** This method does nothing, as we need to use the {@code validate} method instead.
     *
     */
    @Override
    public void run() {}

    /** Performs validation of the network.
     *
     * @param factor The epochs on which we need to print out the validation.
     * @return The validation loss.
     */
    double validate(int factor) {
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
        if (epoch % factor == 0) {
            System.out.printf("Validation loss: %.4f%n", loss);
        }

        if (! isRegressor()) {
            updateCategoricalDataframe();
        }
        // Keep a track of the epoch and increment it here
        epoch ++;
        return loss;
    }

    /** Creates scatter plot from the data and their predicted values.
     *
     * @param title An optional title to append to the plot.
     * @throws IOException If an IO error occurs.
     */
    public void generateScatter(String title) throws IOException {
        super.generateScatter("validation", title);
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
        super.comparisonScatter("validation", title);
    }
}
