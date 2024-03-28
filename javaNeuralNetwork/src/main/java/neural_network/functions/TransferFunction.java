package neural_network.functions;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/** A class to represent the transfer function of a neural network, which
 * creates a linear combination of {@code weights} from {@code values} of
 * {@code Neurons}.
 *
 */
public class TransferFunction implements Activator<List<Double>> {

    private List<Double> weights = new ArrayList<>();
    private double bias = Double.NaN;

    /** Implementation of the transfer function, which returns a sum of
     * w_ij * o_j where w_ij are weights and o_j are values of {@code Neurons}
     * in the left {@code Layer}. Note that we must call {@code bindWeights}
     * before this function.
     *
     * @param o A vector of values from {@code Neurons} in the left
     * @throws IllegalStateException This method should not be called if
     * weights or bias have not been bound.
     * {@code Layer}.
     * @return The output of the transfer function.
     */
    @Override
    public double call(List<Double> o) {
        int o_size = o.size();
        // weights must contain all weights of edges connected to the previous
        // layer from one right neuron
        if (! (weights.size() == o_size)) {
            throw new IllegalStateException("weights must have same size as o," +
                    "check that weights have been bound (%d != %d)"
                            .formatted(weights.size(), o_size));
        } else if (Double.isNaN(bias)) {
            throw new IllegalStateException("bias has not yet been bound, cannot" +
                    "call this method");
        }
        double weightedSum = IntStream
                .iterate(0, j -> j < o_size, j -> j + 1)
                .mapToDouble(j -> o.get(j) * weights.get(j))
                .sum();
        double transferValue = weightedSum + bias;

        // Reset the weights and bias for next time
        weights.clear();
        bias = Double.NaN;
        return transferValue;
    }

    /** Gradient of the transfer function.
     *
     * @param o A vector of values from {@code Neurons} in the left
     *      * {@code Layer}.
     * @return Gradient of transfer function w.r.t weights and bias.
     */
    @Override
    public List<Double> gradient(List<Double> o) {
        // We need to add the bias gradient
        o.add(0.0);
        return o;
    }

    /** Binds {@code weights} from the {@code Edges} to the
     * {@code TransferFunction}.
     *
     * @param weights The new weights to be bound.
     */
    public void bindWeights(List<Double> weights) {
        this.weights.clear();
        this.weights.addAll(weights);
    }

    /** Binds the {@code bias} from the right {@code Neuron} to the
     * {@code TransferFunction}.
     *
     * @param bias The new bias to be bound.
     */
    public void bindBias(double bias) {
        this.bias = bias;
    }
}
