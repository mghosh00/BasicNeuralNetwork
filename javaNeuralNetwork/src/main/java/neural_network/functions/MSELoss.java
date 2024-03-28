package neural_network.functions;

/** Class to test the mean squared error loss for regression networks.
 *
 */
public class MSELoss {

    /** This method simply returns the squared error between the two values
     * {@code yHat} and {@code y}.
     *
     * @param yHat The output value from the {@code Neuron} in the output {@code Layer}.
     * @param y The true label for the datapoint.
     * @return The squared error between the two.
     */
    public static double call(double yHat, double y) {
        return (yHat - y) * (yHat - y);
    }

    /** This method returns the gradient of the loss function.
     *
     * @param yHat The output value from the {@code Neuron} in the output {@code Layer}.
     * @param y The true label for the datapoint.
     * @return The gradient of the squared error between the two.
     */
    public static double gradient(double yHat, double y) {
        return 2 * (yHat - y);
    }
}
