package neural_network.functions;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.exp;

/** Class to represent the sigmoid function, which is an {@code Activator} for
 * {@code Neurons}
 *
 */
public class Sigmoid implements Activator<Double> {

    /** Implementation of sigmoid function.
     *
     * @param x The input value (from the {@code TransferFunction}).
     * @return The sigmoid output.
     */
    @Override
    public double call(Double x) {
        // We do the below to avoid overflow errors
        if (x < 0) {
            return exp(x) /(1 + exp(x));
        }
        return 1 / (1 + exp(-x));
    }

    /**
     *
     * @param x The input value (from the {@code TransferFunction}).
     * @return The gradient of the sigmoid.
     */
    @Override
    public Double gradient(Double x) {
        return call(x) * (1 - call(x));
    }
}
