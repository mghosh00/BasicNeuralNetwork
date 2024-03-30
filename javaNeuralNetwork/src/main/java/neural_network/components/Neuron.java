package neural_network.components;

import java.util.ArrayList;
import java.util.List;

/** Class to represent a single {@code Neuron} in a {@code Network}.
 * @version 1.0.0
 * @since 1.0.0
 */
public class Neuron {

    private final List<Integer> id = new ArrayList<>();
    private double bias = 0.0;
    private double value = Double.NaN;
    private final List<Double> biasGradients = new ArrayList<>();

    /** Constructor method.
     * @param layerId The id of the {@code Layer} of the {@code Network}.
     * @param rowId The row in the {@code Layer}.
     */
    Neuron(int layerId, int rowId) {
        this.id.addAll(List.of(layerId, rowId));
    }

    /** Adds an element to the {@code biasGradients} list.
     *
     * @param biasGradient The bias gradient to be added.
     */
    void addBiasGradient(double biasGradient) {
        biasGradients.add(biasGradient);
    }

    /** Clears the {@code biasGradients} list.
     *
     */
    void clearBiasGradients() {
        biasGradients.clear();
    }

    /** Getter method for {@code id}.
     *
     * @return The {@code id} of form {@code [layer, row]}.
     */
    List<Integer> getId() {
        return id;
    }

    /** Getter method for {@code bias}.
     *
     * @return The {@code bias} of the {@code Neuron}.
     */
    double getBias() {
        return bias;
    }

    /** Setter method for {@code bias}.
     *
     * @param bias The new {@code bias}.
     */
    void setBias(double bias) {
        this.bias = bias;
    }

    /** Getter method for {@code value}.
     *
     * @return The {@code value} of the {@code Neuron}.
     */
    double getValue() {
        return value;
    }

    /** Setter method for {@code value}.
     *
     * @param value The new {@code value}.
     */
    void setValue(double value) {
        this.value = value;
    }

    /** Getter method for {@code biasGradients}.
     *
     * @return A copy of the {@code biasGradients} list.
     */
    List<Double> getBiasGradients() {
        return List.copyOf(biasGradients);
    }

    /** String method.
     *
     * @return {@code String} representation of the {@code Neuron}.
     */
    @Override
    public String toString() {
        return "Neuron %s".formatted(id.toString());
    }
}
