package neural_network.learning;

import neural_network.components.Network;
import neural_network.functions.CrossEntropyLoss;
import neural_network.functions.MSELoss;
import neural_network.util.Header;
import neural_network.util.Partitioner;
import neural_network.util.WeightedPartitioner;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/** Base class for {@code Trainer}, {@code Validator} and {@code Tester}.
 *
 */
public abstract class Learner {

    private Network network;
    private final boolean doRegression;
    private final int dimensions;
    private final int numDatapoints;
    private final int batchSize;
    private List<String> categoryNames;
    private NavigableMap<Header, List<String>> categoricalDf;
    private final NavigableMap<Header, List<Double>> df = new TreeMap<>();
    private CrossEntropyLoss crossEntropyLoss;
    private MSELoss mseLoss;
    private Partitioner partitioner;

    /** General constructor method (with possibility of weighted partition).
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
    public Learner(Network network, NavigableMap<Header, List<String>> data,
                   int batchSize, boolean weighted, int numBins) {
        this.network = network;
        this.doRegression = network.isRegressor();
        data = new TreeMap<>(Map.copyOf(data));

        // Ensure that number of input neurons equals number of features
        int numFeatures = data.size() - 1;
        int numInputNeurons = network.getNeuronCounts().get(0);
        if (numFeatures != numInputNeurons) {
            throw new IllegalArgumentException(
                    "Number of features must match number of " +
                    "initial neurons (features = %d, initial ".formatted(numFeatures) +
                    "neurons = %d)".formatted(numInputNeurons));
        }

        this.dimensions = numFeatures;
        Header.setDimensions(dimensions);
        // Ensure that the batchSize is not too big
        int numDatapoints = data.get(Header.Y).size();
        if (batchSize > numDatapoints) {
            throw new IllegalArgumentException("Batch size must be smaller than " +
                    "number of datapoints");
        }
        this.numDatapoints = numDatapoints;
        this.batchSize = batchSize;

        // This is setting up the df by converting strings to doubles.
        // We do not initialise the Y column here, that is dependent on regression
        // vs classification
        for (Header header : Header.getInitialHeaders().subList(0, numFeatures)) {
            // Convert all strings to doubles
            List<Double> doubleVals = data.get(header).stream()
                    .map(Double::valueOf)
                    .toList();
            df.put(header, doubleVals);
        }
        // Initialise the Y_HAT column with a list of zeros
        df.put(Header.Y_HAT,
                new ArrayList<>(Collections.nCopies(numDatapoints, 0.0)));

        if (doRegression) {
            // If we are doing regression, we have no categories, and we will
            // use a mean squared error loss
            List<Double> doubleVals = data.get(Header.Y).stream()
                    .map(Double::valueOf)
                    .toList();
            df.put(Header.Y, doubleVals);
            this.mseLoss = new MSELoss();
        } else {
            // Save the category names to be used for plots and output data
            categoryNames = new ArrayList<>(new TreeSet<>(data.get(Header.Y)));

            // Ensure that the number of network output neurons equals the number
            // of classes in the dataframe
            int numClasses = categoryNames.size();
            int numLayers = network.getLayers().size();
            int numOutputs = network.getNeuronCounts().get(numLayers - 1);
            if (numOutputs < numClasses) {
                throw new IllegalArgumentException(
                        "The number of output neurons in the " +
                        "network (%d) is less than the ".formatted(numOutputs) +
                        "number of classes in the dataframe " +
                        "(%d)".formatted(numClasses));
            }

            // Change the category names to integers from 0 to numClasses - 1 but
            // save the category names for reference in plots
            data.put(Header.Y_HAT,
                     new ArrayList<>(Collections.nCopies(numDatapoints, "")));
            categoricalDf = new TreeMap<>(data);
            List<Double> yVals = data.get(Header.Y).stream()
                    .map(category -> (double) categoryNames.indexOf(category))
                    .toList();
            df.put(Header.Y, yVals);
            crossEntropyLoss = new CrossEntropyLoss();
        }

        // Now choose the partitioner
        if (weighted) {
            partitioner = new WeightedPartitioner(numDatapoints, batchSize,
                    df.get(Header.Y), doRegression, numBins);
        } else {
            partitioner = new Partitioner(numDatapoints, batchSize);
        }
    }

    /** Constructor for non-weighted partitions.
     *
     * @param network The neural network to train.
     * @param data All the data to be passed to the {@code network}.
     * @param batchSize The number of datapoints per batch for an epoch.
     */
    public Learner(Network network, NavigableMap<Header, List<String>> data,
                   int batchSize) {
        this(network, data, batchSize, false, 10);
    }

    /** Performs the forward pass through the {@code network}
     * for one batch of the data.
     *
     * @param batchIds The random list of ids for the current batch.
     * @return The total loss of the batch (to keep track).
     */
    double forwardPassOneBatch(List<Integer> batchIds) {
        double totalLoss = 0.0;
        for (int id : batchIds) {
            List<Double> labelledPoint = df.keySet().stream()
                    .map(header -> df.get(header).get(id))
                    .toList();
            List<Double> x = labelledPoint.subList(0, dimensions);
            double y = labelledPoint.get(dimensions);

            // Do the forward pass and save the predicted value to the df
            if (doRegression) {
                double yHat = network.forwardPassOneDatapoint(x).get(0);
                totalLoss += mseLoss.call(yHat, y);
                df.get(Header.Y_HAT).set(id, yHat);
            } else {
                int yClass = (int) y;
                // We choose the class with maximal softmax probability as our
                // yHat for output
                List<Double> softmaxVector = network.forwardPassOneDatapoint(x);
                totalLoss += crossEntropyLoss.call(softmaxVector, yClass);
                double yHat = softmaxVector.indexOf(
                        Collections.max(softmaxVector));
                df.get(Header.Y_HAT).set(id, yHat);
            }
            // Store the gradients if this is the training phase
            storeGradients(id);
        }
        // Return the total loss for this batch
        return totalLoss;
    }

    /** To be overridden by a {@code Trainer}, but will not be touched by the
     * {@code Validator} or {@code Tester}.
     *
     * @param id The id of the current datapoint.
     */
    void storeGradients(int id) {}

    /** Performs training/validation/testing
     *
     */
    public abstract void run();

    /** Update the categorical dataframe with {@code yHat} data but using the
     * original categories from the data - to be used for plotting and
     * outputs to the user. Note that this method will be called after
     * training/testing/validation is complete so that the {@code yHat} values are
     * fully updated.
     *
     */
    void updateCategoricalDataframe() {
        if (doRegression) {
            throw new RuntimeException("Cannot call updateCategoricalDataframe " +
                    "with a regression network");
        }
        // Below is how we keep track of the index of the datapoint
        // and increment it for each element of the stream
        AtomicInteger index = new AtomicInteger();

        // Here we convert each of the integer categories to string named categories
        // and update the categoricalDf with these new values
        df.get(Header.Y_HAT).stream()
                .mapToInt(x -> (int) (double) x)
                .forEach(category -> {
                    categoricalDf.get(Header.Y_HAT).set(index.get(), categoryNames.get(category));
                    index.getAndIncrement();
                });
    }

    /** Getter for {@code network}. For subclasses.
     *
     * @return The {@code network}.
     */
    Network getNetwork() {
        return network;
    }

    /** Getter for {@code doRegression}. For subclasses.
     *
     * @return If {@code true} then we are regressing, else we are classifying.
     */
    boolean isRegressor() {
        return doRegression;
    }

    /** Getter for {@code numDatapoints}. For subclasses.
     *
     * @return The {@code numDatapoints}.
     */
    int getNumDatapoints() {
        return numDatapoints;
    }

    /** Getter for {@code batchSize}. For subclasses.
     *
     * @return The {@code batchSize}.
     */
    int getBatchSize() {
        return batchSize;
    }

    /** Getter for {@code categoryNames}. For subclasses.
     *
     * @return A copy of the {@code categoryNames}.
     */
    List<String> getCategoryNames() {
        return List.copyOf(categoryNames);
    }

    /** Getter for {@code categoricalDf}. For subclasses.
     *
     * @return A deep copy of the {@code categoricalDf}.
     */
    NavigableMap<Header, List<String>> getCategoricalDf() {
        NavigableMap<Header, List<String>> returnMap = new TreeMap<>();
        for (Header header : categoricalDf.keySet()) {
            returnMap.put(header, List.copyOf(categoricalDf.get(header)));
        }
        return returnMap;
    }

    /** Getter for {@code df}. For subclasses.
     *
     * @return A deep copy of the {@code df}.
     */
    NavigableMap<Header, List<Double>> getDf() {
        NavigableMap<Header, List<Double>> returnMap = new TreeMap<>();
        for (Header header : df.keySet()) {
            returnMap.put(header, List.copyOf(df.get(header)));
        }
        return returnMap;
    }

    /** Getter for {@code partitioner}. For subclasses.
     *
     * @return The {@code partitioner}.
     */
    Partitioner getPartitioner() {
        return partitioner;
    }

    /** Setter for {@code crossEntropyLoss}. For mocking.
     *
     * @param crossEntropyLoss The new {@code crossEntropyLoss}.
     */
    void setCrossEntropyLoss(CrossEntropyLoss crossEntropyLoss) {
        this.crossEntropyLoss = crossEntropyLoss;
    }

    /** Setter for {@code mseLoss}. For mocking.
     *
     * @param mseLoss The new {@code mseLoss}.
     */
    void setMseLoss(MSELoss mseLoss) {
        this.mseLoss = mseLoss;
    }

    /** Setter for {@code network}. For mocking.
     *
     * @param network The new {@code network}.
     */
    void setNetwork(Network network) {
        this.network = network;
    }

    /** Setter for {@code partitioner}. For mocking.
     *
     * @param partitioner The new {@code partitioner}.
     */
    void setPartitioner(Partitioner partitioner) {
        this.partitioner = partitioner;
    }

    /** Setter for {@code yHat} of the categorical dataframe. For testing.
     *
     * @param yHat The new {@code yHat}.
     */
    void setYHat(List<String> yHat) {
        categoricalDf.get(Header.Y_HAT).clear();
        categoricalDf.get(Header.Y_HAT).addAll(yHat);
    }
}
