package neural_network.components;

import neural_network.functions.LeakyReLU;
import neural_network.functions.MSELoss;
import neural_network.functions.Softmax;
import neural_network.functions.TransferFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static java.lang.Math.sqrt;

/** Class to represent the whole network.
 *
 */
public class Network {

    // Numbers of neurons
    private final int numFeatures;
    private final int numHiddenLayers;
    private final List<Integer> neuronCounts = new ArrayList<>();

    // Regression boolean
    private final boolean regression;

    // Layer info
    private final Layer inputLayer;
    private final List<Layer> hiddenLayers = new ArrayList<>();
    private final Layer outputLayer;
    private final List<Layer> layers = new ArrayList<>();

    // Edge info
    private final List<List<List<Edge>>> edges = new ArrayList<>();

    // Functions
    private final TransferFunction transfer;
    private final LeakyReLU relu;
    private MSELoss mseLoss = null;
    private Softmax softmax = null;

    // Hyperparameters
    private final double learningRate;
    private final boolean adaptive;
    private final double gamma;
    private static Random random = new Random();

    /** Minimal constructor - need to specify number of inputs, outputs and
     * hidden layers.
     *
     * @param numFeatures The number of coordinates per datapoint.
     * @param numHiddenLayers The number of hidden {@code Layers} for the {@code Network}.
     * @param neuronCounts The number of {@code Neurons} in each hidden {@code Layer}.
     */
    public Network(int numFeatures, int numHiddenLayers, List<Integer> neuronCounts) {
        this(numFeatures, numHiddenLayers, neuronCounts, 2,0.01,
                0.01, false, false, Double.NaN,
                false);
    }

    /** Medium level constructor - for classification.
     *
     * @param numFeatures The number of coordinates per datapoint.
     * @param numHiddenLayers The number of hidden {@code Layers} for the {@code Network}.
     * @param neuronCounts The number of {@code Neurons} in each hidden {@code Layer}.
     * @param numClasses The number of classes for a classification task.
     * @param leak The leak for the {@code LeakyReLU}.
     * @param learningRate The learning rate of the {@code Network}.
     */
    public Network(int numFeatures, int numHiddenLayers, List<Integer> neuronCounts,
                   int numClasses, double leak, double learningRate) {
        this(numFeatures, numHiddenLayers, neuronCounts, numClasses, leak,
                learningRate, false, false, Double.NaN, false);
    }

    /** Most general constructor.
     *
     * @param numFeatures The number of coordinates per datapoint.
     * @param numHiddenLayers The number of hidden {@code Layers} for the {@code Network}.
     * @param neuronCounts The number of {@code Neurons} in each hidden {@code Layer}.
     * @param numClasses The number of classes for a classification task.
     * @param leak The leak for the {@code LeakyReLU}.
     * @param learningRate The learning rate of the {@code Network}.
     * @param regression Whether we are performing regression or not ({@code false} for
     *                   classification).
     * @param adaptive Whether we wish to have an adaptive {@code learningRate} or not.
     * @param gamma The adaptive learning rate parameter.
     * @param heWeights Whether we wish to use He initialisation for the weights or not.
     */
    public Network(int numFeatures, int numHiddenLayers, List<Integer> neuronCounts,
                   int numClasses, double leak, double learningRate, boolean regression,
                   boolean adaptive, double gamma, boolean heWeights) {
        if (numHiddenLayers != neuronCounts.size()) {
            throw new IllegalArgumentException(
                    "neuronCounts (%d) must have a length equal to numHiddenLayers (%d)"
                            .formatted(neuronCounts.size(), numHiddenLayers));
        }

        // Initial set up
        this.numFeatures = numFeatures;
        this.numHiddenLayers = numHiddenLayers;
        this.neuronCounts.addAll(neuronCounts);
        this.regression = regression;

        // If we are doing regression, set numClasses to 1 no matter the input value
        if (regression) {
            numClasses = 1;
        }

        // Layers
        this.inputLayer = new Layer(0, numFeatures);
        this.hiddenLayers.addAll(Stream
                .iterate(1, j -> j <= numHiddenLayers, j -> j + 1)
                .map(j -> new Layer(j, neuronCounts.get(j - 1)))
                .toList());
        this.outputLayer = new Layer(numHiddenLayers + 1, numClasses);
        this.layers.add(inputLayer);
        this.layers.addAll(hiddenLayers);
        this.layers.add(outputLayer);

        // Edges
        // Order in edges is left layer then right neuron then left neuron
        for (int i = 0; i <= numHiddenLayers; i ++) {
            Layer leftLayer = layers.get(i);
            Layer rightLayer = layers.get(i + 1);
            List<List<Edge>> layerList = new ArrayList<>();
            for (Neuron rightNeuron : rightLayer.getNeurons()) {
                List<Edge> edgeList = leftLayer.getNeurons().stream()
                        .map(leftNeuron -> new Edge(leftNeuron, rightNeuron))
                        .toList();
                layerList.add(edgeList);

                // If we are using He initialisation, we will set the weights here
                // according to that (mean = 0, std dev = 2 / leftLayer.size())
                if (heWeights) {
                    int n = leftLayer.size();
                    for (Edge edge : edgeList) {
                        edge.setWeight(random.nextGaussian(0.0,
                                sqrt(2.0 / n)));
                    }
                }
            }
            edges.add(layerList);
        }

        // Functions
        this.transfer = new TransferFunction();
        this.relu = new LeakyReLU(leak);
        if (regression) {
            this.mseLoss = new MSELoss();
        } else {
            this.softmax = new Softmax();
        }

        // Hyper-parameters
        this.learningRate = learningRate;
        this.adaptive = adaptive;
        this.gamma = gamma;
    }

    /** Performs a forward pass for one datapoint, excluding the ground
     * truth value. This method returns the predicted value in the final
     * {@code Neuron} in the {@code outputLayer}.
     *
     * @param x The datapoint, with all features.
     * @return The softmax probabilities of each class (for classification) or
     *             the predicted regression value (for regression).
     */
    public List<Double> forwardPassOneDatapoint(List<Double> x) {
        if (x.size() != inputLayer.size()) {
            throw new IllegalArgumentException(
                    "Number of features must match the number of neurons in the input layer " +
                            "(%d != %d)".formatted(x.size(), inputLayer.size()));
        }

        // Input layer
        List<Neuron> inputNeurons = inputLayer.getNeurons();
        for (int j = 0; j < inputLayer.size(); j ++) {
            inputNeurons.get(j).setValue(x.get(j));
        }

        // This is setting up for the output layer
        List<Double> zOutputLayer = new ArrayList<>();
        // Hidden layers and output layer
        for (Layer leftLayer : layers.subList(0, layers.size() - 1)) {
            Layer rightLayer = layers.get(leftLayer.getId() + 1);

            if (rightLayer.getId() != outputLayer.getId()) {
                for (Neuron rightNeuron : rightLayer.getNeurons()) {
                    // Calculates the desired value for each neuron
                    double z = calculatePreActivatedValue(leftLayer, rightNeuron);
                    // Uses ReLU activation for the neuron
                    rightNeuron.setValue(relu.call(z));
                }
            } else {
                if (regression) {
                    // We only have one output neuron with linear activation
                    // for a regression network
                    Neuron outputNeuron = outputLayer.getNeurons().get(0);
                    double z = calculatePreActivatedValue(leftLayer, outputNeuron);
                    outputNeuron.setValue(z);
                    // Here, we exit the method, returning the value of the
                    // output neuron
                    return new ArrayList<>(List.of(z));
                } else {
                    for (Neuron rightNeuron : rightLayer.getNeurons()) {
                        double z = calculatePreActivatedValue(leftLayer, rightNeuron);
                        zOutputLayer.add(z);
                    }
                }
            }
        }

        // Output layer
        // Activates the output layer using softmax activation
        return activateOutputLayer(zOutputLayer);
    }

    /** Given a {@code leftLayer} and a {@code rightNeuron}, this calculates the
     * activation function and value from the {@code leftLayer} and propagates
     * this value to the {@code rightNeuron}.
     *
     * @param leftLayer The current left {@code Layer} in forward propagation.
     * @param rightNeuron The current right {@code Neuron} in forward propagation.
     * @return The value returned by the {@code transferFunction} before activation.
     */
    double calculatePreActivatedValue(Layer leftLayer, Neuron rightNeuron) {
        List<Neuron> leftNeurons = leftLayer.getNeurons();
        int i = rightNeuron.getId().get(0); int j = rightNeuron.getId().get(1);

        // All edges connecting the leftLayer to the rightNeuron
        List<Edge> innerEdges = edges.get(i - 1).get(j);

        // Lists of values and weights, with the bias
        List<Double> oList = leftNeurons.stream()
                .map(Neuron::getValue)
                .toList();
        List<Double> weights = innerEdges.stream()
                .map(Edge::getWeight)
                .toList();
        double bias = rightNeuron.getBias();
        transfer.bindWeights(weights);
        transfer.bindBias(bias);

        // Return the pre-activated value for this neuron
        return transfer.call(oList);
    }

    /** Activates the values from the {@code outputLayer} using the {@code softmax}
     * activation function.
     *
     * @param zOutputLayer The pre-activated values from the output layer in a
     *                     classification problem.
     * @return The list of softmax probabilities.
     */
    List<Double> activateOutputLayer(List<Double> zOutputLayer) {
        softmax.normalise(zOutputLayer);
        List<Neuron> softmaxNeurons = outputLayer.getNeurons();
        List<Double> softmaxVector = new ArrayList<>();
        for (int j = 0; j < zOutputLayer.size(); j ++) {
            double softmaxValue = softmax.call(zOutputLayer.get(j));
            softmaxVector.add(softmaxValue);
            softmaxNeurons.get(j).setValue(softmaxValue);
        }
        return softmaxVector;
    }

    /** Calculates the gradient of the loss function with respect to one
     * weight (assigned to the {@code edge}) based on the values at edges of future
     * layers. One part of the back propagation process.
     *
     * @param edge The {@code Edge} containing the weight we are interested in.
     * @param target The target value for the final output neuron for this specific
     *               datapoint.
     * @param first Determines whether we find the bias gradient or not.
     */
    public void storeGradientOfLoss(Edge edge, double target, boolean first) {
        int leftLayerIndex = edge.getId().get(0);
        Neuron rightNeuron = edge.getRightNeuron();

        // Value of left neuron and right neuron
        double oLeft = edge.getLeftNeuron().getValue();
        double oRight = rightNeuron.getValue();

        int rightIndex = rightNeuron.getId().get(0); int row = rightNeuron.getId().get(1);

        // Output layer
        if (leftLayerIndex == numHiddenLayers) {
            double delta = (regression) ? mseLoss.gradient(oRight, target)
                    : oRight - Boolean.compare(row == (int) target, false);
            edge.setDelta(delta);
            edge.addLossGradient(oLeft * delta);

            // If this is the first time we call this function for the neuron,
            // we need to also store the gradient of the loss for the bias
            if (first) {
                rightNeuron.addBiasGradient(delta);
            }
        } else {
            // Hidden layers
            Layer nextLayer = layers.get(rightIndex + 1);

            // Edges connected to the current rightNeuron in the layer to
            // the right of the right layer
            List<Edge> nextEdges = Stream
                    .iterate(0, j -> j < nextLayer.size(), j -> j + 1)
                    .map(j -> edges.get(rightIndex).get(j).get(row))
                    .toList();

            double factor = nextEdges.stream()
                    .mapToDouble(newEdge -> newEdge.getWeight() * newEdge.getDelta())
                    .sum();

            // Constant (either +1 or self._leak)
            double reluGrad = relu.gradient(oRight);
            double delta = factor * reluGrad;
            edge.setDelta(delta);
            edge.addLossGradient(oLeft * delta);

            if (first) {
                rightNeuron.addBiasGradient(delta);
            }
        }
    }

    /** Uses the loss gradients of all datapoints (for this specific {@code edge})
     * to perform gradient descent and calculate a new weight for this {@code edge}.
     *
     * @param edge The {@code Edge} whose weight we are interested in updating.
     */
    public void backPropagateWeight(Edge edge) {
        double currentWeight = edge.getWeight();

        // The number of datapoints which we have passed through the network
        // in a batch
        int batchSize = edge.getLossGradients().size();
        double avgLossGradient = edge.getLossGradients().stream()
                .mapToDouble(a -> a)
                .sum() / batchSize;

        // If the network is adaptive we need a different gradient descent algorithm
        if (adaptive) {
            double velocity = gamma * edge.getVelocity() +
                    learningRate * avgLossGradient;
            edge.setWeight(currentWeight - velocity);
            edge.setVelocity(velocity);
        } else {
            edge.setWeight(currentWeight - learningRate * avgLossGradient);
        }

        // Reset for the next batch
        edge.clearLossGradients();
    }

    /** Uses the bias gradients of all datapoints (for this specific {@code Neuron})
     * to perform gradient descent and calculate a new bias for this {@code Neuron}.
     *
     * @param neuron The {@code Neuron} whose bias we are interested in updating.
     */
    public void backPropagateBias(Neuron neuron) {
        double currentBias = neuron.getBias();
        double batchSize = neuron.getBiasGradients().size();
        double avgBiasGradient = neuron.getBiasGradients().stream()
                .mapToDouble(a -> a)
                .sum() / batchSize;
        neuron.setBias(currentBias - learningRate * avgBiasGradient);
        neuron.clearBiasGradients();
    }

    /** Getter method for all the neuron counts, for the input,
     * hidden and output {@code Layers}.
     *
     * @return All neuron counts.
     */
    public List<Integer> getNeuronCounts() {
        List<Integer> allNeuronCounts = new ArrayList<>();
        allNeuronCounts.add(numFeatures);
        allNeuronCounts.addAll(neuronCounts);
        allNeuronCounts.add(outputLayer.size());
        return allNeuronCounts;
    }

    /** Getter method for the {@code regression} parameter.
     *
     * @return {@code true} if we are doing regression, {@code false} otherwise.
     */
    public boolean isRegressor() {
        return regression;
    }

    /** Getter method for the {@code layers}.
     *
     * @return A list of all the layers of the {@code Network}.
     */
    public List<Layer> getLayers() {
        return List.copyOf(layers);
    }

    /** Getter method for the {@code edges}.
     *
     * @return A list of all the edges of the {@code Network}.
     */
    public List<List<List<Edge>>> getEdges() {
        List<List<List<Edge>>> copyEdges = new ArrayList<>();
        for (List<List<Edge>> list1 : edges) {
            List<List<Edge>> copyEdges1 = new ArrayList<>();
            for (List<Edge> list2 : list1) {
                copyEdges1.add(List.copyOf(list2));
            }
            copyEdges.add(copyEdges1);
        }
        return copyEdges;
    }

    /** Setter method for {@code random}.
     *
     * @param random The new {@code random}. This can be used for mocking
     *               purposes.
     */
    static void setRandom(Random random) {
        Network.random = random;
    }
}
