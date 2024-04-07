package neural_network.learning;

import neural_network.components.Edge;
import neural_network.components.Network;
import neural_network.util.Header;
import neural_network.util.Plotter;

import java.io.IOException;
import java.util.*;

/** Class to train a neural network.
 *
 */
public class Trainer extends Learner {

    private final int numEpochs;
    private Validator validator;
    private final Map<String, List<Double>> lossDf = new TreeMap<>();

    /** General constructor method.
     *
     * @param network The neural network to train.
     * @param data All the training data for the {@code network}.
     * @param batchSize The number of datapoints used in each epoch.
     * @param weighted If {@code true} then we use the {@code WeightedPartitioner},
     *                 otherwise we use the standard {@code Partitioner}.
     * @param numBins If {@code weighted} is {@code true} and the {@code network} is
     *                a regressor, then we need to specify the number of bins for the
     *                {@code WeightedPartitioner}. Otherwise, this parameter is ignored.
     * @param numEpochs The number of epochs we are training for.
     * @param validator The validator used (if any). Pass {@code null}
     *                  for no validation.
     */
    public Trainer(Network network, NavigableMap<Header, List<String>> data,
                   int batchSize, boolean weighted, int numBins, int numEpochs,
                   Validator validator) {
        super(network, data, batchSize, weighted, numBins);
        this.numEpochs = numEpochs;
        this.validator = validator;
        this.lossDf.put("Training", new ArrayList<>());
        if (validator != null) {
            this.lossDf.put("Validation", new ArrayList<>());
        }
    }

    /** Defaults constructor. Used for unweighted partitioner and no validation.
     *
     * @param network The neural network to train.
     * @param data All the training data for the {@code network}.
     * @param batchSize The number of datapoints used in each epoch.
     * @param numEpochs The number of epochs we are training for.
     */
    public Trainer(Network network, NavigableMap<Header, List<String>> data,
                   int batchSize, int numEpochs) {
        super(network, data, batchSize);
        this.numEpochs = numEpochs;
        this.lossDf.put("Training", new ArrayList<>());
    }

    /** Stores the gradients of the loss functions after a forward pass of
     * a single datapoint.
     *
     * @param id The id of the current datapoint.
     */
    @Override
    void storeGradients(int id) {
        double y = getDf().get(Header.Y).get(id);

        // Take gradients of loss and store them in the edges (backwards)
        List<List<List<Edge>>> edges = getNetwork().getEdges();
        Collections.reverse(edges);
        for (List<List<Edge>> edgeLayer : edges) {
            for (List<Edge> rightNeuron : edgeLayer) {
                boolean first = true;
                for (Edge edge : rightNeuron) {
                    getNetwork().storeGradientOfLoss(edge, y, first);
                    first = false;
                }
            }
        }
    }

    /** Performs back propagation for one batch of datapoints (stored within
     * the memory of the edges).
     *
     */
    void backPropagateOneBatch() {
        // Back propagate all the weights first
        getNetwork().backPropagateWeights();

        // Then back propagate all the biases
        getNetwork().backPropagateBiases();
    }

    /** Performs training of the network.
     *
     */
    @Override
    public void run() {
        int factor = (int) Math.ceil((double) numEpochs / 100);
        for (int epoch = 0; epoch < numEpochs; epoch ++) {
            double totalLoss = 0.0;
            // Partition all the datapoints into batches
            List<List<Integer>> batchPartition = getPartitioner().call();
            int itsPerEpoch = (int) Math
                    .ceil((double) getNumDatapoints() / getBatchSize());
            for (int iteration = 0; iteration < itsPerEpoch; iteration ++) {
                List<Integer> batchIds = batchPartition.get(iteration);

                // Do forward pass and back propagation for this specific batch
                totalLoss += forwardPassOneBatch(batchIds);
                backPropagateOneBatch();
            }
            double loss = Math.round(10000 * totalLoss / getNumDatapoints()) / 10000.0;
            if (epoch % factor == 0) {
                System.out.println("Epoch: " + epoch);
                System.out.printf("Training loss: %.4f%n", loss);
            }
            // Record the loss and potential validation loss
            lossDf.get("Training").add(loss);
            if (validator != null) {
                double validationLoss = validator.validate(factor);
                lossDf.get("Validation").add(validationLoss);
            }
        }
        // At the end, update the categorical dataframe
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
        super.generateScatter("training", title);
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
        super.comparisonScatter("training", title);
    }

    /** Creates a plot of the training and validation loss over time.
     *
     * @param title An optional title to append to the plot.
     * @throws IOException If an IO error occurs.
     */
    public void generateLossPlot(String title) throws IOException {
        Plotter.plotLoss(lossDf, title);
    }

    /** Getter for {@code lossDf}.
     *
     * @return A deep copy of the {@code lossDf}.
     */
    public Map<String, List<Double>> getLossDf() {
        Map<String, List<Double>> returnMap = new HashMap<>();
        for (String header : lossDf.keySet()) {
            returnMap.put(header, List.copyOf(lossDf.get(header)));
        }
        return returnMap;
    }
}
