package neural_network.functions;

/** Class to represent a leaky ReLU function (can be just a ReLU if the {@code leak}
 * is set to 0.0), which is an {@code Activator} for {@code Neurons}.
 *
 */
public class LeakyReLU implements Activator<Double> {

    private final double leak;

    /** Constructor method with a given {@code leak}.
     *
     * @param leak The leak for a leaky ReLU function.
     */
    public LeakyReLU(double leak) {
        this.leak = leak;
    }

    /** No-args constructor method - {@code leak} defaults to zero. This
     * represents a standard ReLU function.
     *
     */
    public LeakyReLU() {
        this(0.0);
    }

    /** Implementation of the leaky ReLU function.
     *
     * @param x The input value (from the {@code TransferFunction}).
     * @return The leaky ReLU output.
     */
    @Override
    public double call(Double x) {
        return x >= 0 ? x : x * leak;
    }

    /** The gradient of the leaky ReLU function.
     *
     * @param x The input value (from the {@code TransferFunction}).
     * @return The gradient of the leaky ReLU.
     */
    @Override
    public Double gradient(Double x) {
        return x >= 0 ? 1 : leak;
    }

    /** Getter method for {@code leak}.
     *
     * @return The leak of the leaky ReLU.
     */
    double getLeak() {
        return leak;
    }
}
