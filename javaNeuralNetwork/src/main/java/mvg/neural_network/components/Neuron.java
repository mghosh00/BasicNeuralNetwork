package mvg.neural_network.components;

import java.util.ArrayList;
import java.util.List;

/** Class to represent a single Neuron in a neural network.
 * @version 1.0.0
 * @since 1.0.0
 */
class Neuron {

    private final List<Integer> id = new ArrayList<>();
    private double bias = 0.0;
    private double value = Double.NaN;
    private final List<Double> biasGradients = new ArrayList<>();

    /** Constructor method.
     * @param layerId The id of the layer of the network.
     * @param rowId The row in the layer.
     */
    Neuron(int layerId, int rowId) {
        this.id.addAll(List.of(layerId, rowId));
    }

    /** Adds an element to the biasGradients list.
     *
     * @param biasGradient The bias gradient to be added.
     */
    void addBiasGradient(double biasGradient) {
        biasGradients.add(biasGradient);
    }

    /** Clears the biasGradients list.
     *
     */
    void clearBiasGradients() {
        biasGradients.clear();
    }

    /** Getter method for id.
     *
     * @return The id.
     */
    List<Integer> getId() {
        return id;
    }

    /** Getter method for bias.
     *
     * @return The bias.
     */
    double getBias() {
        return bias;
    }

    /** Setter method for bias.
     *
     * @param bias The new bias.
     */
    void setBias(double bias) {
        this.bias = bias;
    }

    /** Getter method for value.
     *
     * @return The value.
     */
    double getValue() {
        return value;
    }

    /** Setter method for value.
     *
     * @param value The new value.
     */
    void setValue(double value) {
        this.value = value;
    }

    /** Getter method for biasGradients.
     *
     * @return A copy of the biasGradients list.
     */
    List<Double> getBiasGradients() {
        return List.copyOf(biasGradients);
    }

    /** String method.
     *
     * @return String representation of the Neuron.
     */
    @Override
    public String toString() {
        return "Neuron %s".formatted(id.toString());
    }
}
