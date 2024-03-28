package neural_network.functions;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static java.lang.Math.exp;

/** Class to represent a softmax function. This takes in the vector of values
 * from the output neurons in a classification problem and converts them into
 * softmax probabilities. This is an {@code Activator} function for the output
 * {@code Layer}.
 *
 */
public class Softmax implements Activator<Double> {

    private double normalisation = 1.0;
    private double maxZ = 0.0;

    /** Calculates the normalisation constant for the softmax function.
     * We wish to avoid any overflow errors, so we multiply the normalisation
     * constant by exp(m). We account for this when finding the
     * Softmax value later.
     *
     * @param zList The vector of pre-activated {@code values} from the output
     *              {@code Layer}.
     */
    public void normalise(List<Double> zList) {
        maxZ = Collections.max(zList);
        normalisation = Stream
                .iterate(0, j -> j < zList.size(), j -> j + 1)
                .mapToDouble(j -> exp(zList.get(j) - maxZ))
                .sum();
    }

    /** The softmax function. Note we multiply top and bottom by _max_z to
     * avoid any overflow error.
     *
     * @param z The value of an output neuron.
     * @return The softmax activation value of {@code z}
     */
    @Override
    public double call(Double z) {
        return exp(z - maxZ) / normalisation;
    }

    /** Note that the gradient of {@code Softmax} is dealt with elsewhere.
     * Therefore, we throw an exception if there is an attempt to call this
     * method.
     *
     * @param z The value of an output neuron.
     * @throws UnsupportedOperationException This method should not be called.
     */
    @Override
    public Double gradient(Double z) {
        throw new UnsupportedOperationException("gradient method should not be" +
                "called from the Softmax class.");
    }
}
