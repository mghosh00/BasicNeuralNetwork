package neural_network.components;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Class to represent an {@code Edge} joining two {@code Nodes} of a network
 *
 */
public class Edge {

    private final Neuron leftNeuron;
    private final Neuron rightNeuron;
    private final List<Integer> id = new ArrayList<>();
    private double weight;
    private final List<Double> lossGradients = new ArrayList<>();
    private double delta = 0.0;
    private double velocity = 0.0;
    private static Random random = new Random();

    /** Constructor method.
     * @param leftNeuron The left {@code Neuron} of the {@code Edge}.
     * @param rightNeuron The right {@code Neuron} of the {@code Edge}.
     */
    Edge(Neuron leftNeuron, Neuron rightNeuron) {
        this.leftNeuron = leftNeuron;
        this.rightNeuron = rightNeuron;
        int leftLayerId = leftNeuron.getId().get(0);
        int leftRowId = leftNeuron.getId().get(1);
        int rightLayerId = rightNeuron.getId().get(0);
        int rightRowId = rightNeuron.getId().get(1);
        if (leftLayerId + 1 != rightLayerId) {
            throw new RuntimeException("Edge must connect adjacent layers (left: %d, right: %d)"
                    .formatted(leftLayerId, rightLayerId));
        }
        // Below is a three-membered id to uniquely determine the edge
        this.id.addAll(List.of(leftLayerId, leftRowId, rightRowId));
        this.weight = random.nextDouble(-1, 1);
    }

    /** Adds an element to the {@code lossGradients} list.
     *
     * @param lossGradient The loss gradient to be added.
     */
    void addLossGradient(double lossGradient) {
        lossGradients.add(lossGradient);
    }

    /** Clears the {@code lossGradients} list.
     *
     */
    void clearLossGradients() {
        lossGradients.clear();
    }

    /** Getter method for {@code leftNeuron}.
     *
     * @return The {@code leftNeuron}.
     */
    Neuron getLeftNeuron() {
        return leftNeuron;
    }

    /** Getter method for {@code rightNeuron}.
     *
     * @return The {@code rightNeuron}.
     */
    Neuron getRightNeuron() {
        return rightNeuron;
    }

    /** Getter method for {@code id}.
     *
     * @return The {@code id}.
     */
    List<Integer> getId() {
        return id;
    }

    /** Getter method for {@code weight}.
     *
     * @return The {@code weight}.
     */
    double getWeight() {
        return weight;
    }

    /** Setter method for {@code weight}.
     *
     * @param weight The new {@code weight}.
     */
    void setWeight(double weight) {
        this.weight = weight;
    }

    /** Getter method for {@code lossGradients}.
     *
     * @return The {@code lossGradients}.
     */
    List<Double> getLossGradients() {
        return List.copyOf(lossGradients);
    }

    /** Getter method for {@code delta}.
     *
     * @return The {@code delta}.
     */
    double getDelta() {
        return delta;
    }

    /** Setter method for {@code delta}.
     *
     * @param delta The new {@code delta}.
     */
    void setDelta(double delta) {
        this.delta = delta;
    }

    /** Getter method for {@code velocity}.
     *
     * @return The {@code velocity}.
     */
    double getVelocity() {
        return velocity;
    }

    /** Setter method for {@code velocity}.
     *
     * @param velocity The new {@code velocity}.
     */
    void setVelocity(double velocity) {
        this.velocity = velocity;
    }

    /** Setter method for {@code random}.
     *
     * @param random The new {@code random}. This can be used for mocking
     *               purposes.
     */
    static void setRandom(Random random) {
        Edge.random = random;
    }

    /** String method.
     *
     * @return {@code String} representation of the {@code Edge}.
     */
    @Override
    public String toString() {
        return "Edge %s".formatted(id.toString());
    }
}
