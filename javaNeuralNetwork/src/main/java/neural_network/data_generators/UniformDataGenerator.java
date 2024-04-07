package neural_network.data_generators;

import java.lang.reflect.Method;
import java.util.List;
import java.util.stream.Stream;

/** Class to randomly generate datapoints and categorise them according to
 * a given rule, or provide an output value if we are regressing,
 * with data being generated via a uniform distribution.
 *
 * @param <T> The return type of the {@code function} (could be any type
 *            for a classification problem).
 */
public class UniformDataGenerator<T> extends DataGenerator<T> {

    private final List<Double> lowerBounds;
    private final List<Double> upperBounds;

    /**
     * Constructor method for {@code NormalDataGenerator}.
     *
     * @param function      A rule which takes a certain number of coordinates and returns a
     *                      value representing the class or function output of the datapoint.
     *                      This must be a static method from the underlying class.
     * @param numDatapoints The number of datapoints to be generated.
     * @param lowerBounds A list of lower bounds for each coordinate.
     * @param upperBounds A list of upper bounds for each coordinate.
     * @throws IllegalArgumentException If the parameter types of {@code function}
     *                                  are not {@code doubles}, if the number of coordinates are invalid or if
     *                                  {@code numDatapoints} is not greater than {@code 0}.
     */
    public UniformDataGenerator(Method function, int numDatapoints,
                                List<Double> lowerBounds, List<Double> upperBounds) {
        super(function, numDatapoints);
        if (lowerBounds.size() != getDimensions()) {
            throw new IllegalArgumentException("The function method accepts " +
                    "%d parameters but we have %d lower bounds."
                            .formatted(getDimensions(), lowerBounds.size()));
        }
        if (upperBounds.size() != getDimensions()) {
            throw new IllegalArgumentException("The function method accepts " +
                    "%d parameters but we have %d upper bounds."
                            .formatted(getDimensions(), upperBounds.size()));
        }
        for (int i = 0; i < lowerBounds.size(); i ++) {
            if (lowerBounds.get(i) >= upperBounds.get(i)) {
                throw new IllegalArgumentException ("All lower bounds must be lower " +
                        "than their related upper bounds (%f >= %f)"
                                .formatted(lowerBounds.get(i), upperBounds.get(i)));
            }
        }
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
    }

    /** Generates uniformly distributed data.
     *
     */
    @Override
    void generateData() {
        List<List<Double>> xData = Stream
                .iterate(0, i -> i < getDimensions(), i -> i + 1)
                .map(i -> Stream
                        .generate(() -> getRandom().nextDouble(lowerBounds.get(i), upperBounds.get(i)))
                        .limit(getNumDatapoints())
                        .toList())
                .toList();
        addData(xData);
    }
}
