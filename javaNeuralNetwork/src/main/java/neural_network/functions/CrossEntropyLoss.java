package neural_network.functions;

import java.util.List;

import static java.lang.Math.log;

/** Class to represent the cross entropy loss function for classification
 * networks.
 * This sees how close the {@code Softmax} probabilities are to the true
 * labels of the datapoints.
 */
public class CrossEntropyLoss {

    /** This returns the cross entropy loss, which is the negative logarithm of
     * the probability of the true label.
     *
     * @param yHat The list of {@code Softmax} probabilities.
     * @param y The index of the true label of the datapoint.
     * @throws IllegalArgumentException Thrown if the chosen {@code yHat} value
     * is not an acceptable probability.
     * @return The cross entropy loss.
     */
    public static double call(List<Double> yHat, int y) {
        double softmaxValue = yHat.get(y);
        if (0 <= softmaxValue && softmaxValue <= 1) {
            return -log(softmaxValue);
        } else {
            throw new IllegalArgumentException("Softmax value should be between 0" +
                    " and 1 (yHat.get(%d) = %f)".formatted(y, softmaxValue));
        }
    }
}
