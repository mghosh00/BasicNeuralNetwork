package neural_network.data_generators;

import java.lang.reflect.Method;
import java.util.List;
import java.util.stream.Stream;

/**
 *
 * @param <T>
 */
public class NormalDataGenerator<T> extends DataGenerator<T> {

    private final List<Double> means;
    private final List<Double> stdDevs;

    /**
     * Constructor method for {@code NormalDataGenerator}.
     *
     * @param function      A rule which takes a certain number of coordinates and returns a
     *                      value representing the class or function output of the datapoint.
     *                      This must be a static method from the underlying class.
     * @param numDatapoints The number of datapoints to be generated.
     * @param means A list of means for each coordinate.
     * @param stdDevs A list of standard deviations for each coordinate.
     * @throws IllegalArgumentException If the parameter types of {@code function}
     *                                  are not {@code doubles}, if the number of coordinates are invalid or if
     *                                  {@code numDatapoints} is not greater than {@code 0}.
     */
    public NormalDataGenerator(Method function, int numDatapoints,
                               List<Double> means, List<Double> stdDevs) {
        super(function, numDatapoints);
        if (means.size() != getDimensions()) {
            throw new IllegalArgumentException("The function method accepts " +
                    "%d parameters but we have %d means."
                            .formatted(getDimensions(), means.size()));
        }
        if (stdDevs.size() != getDimensions()) {
            throw new IllegalArgumentException("The function method accepts " +
                    "%d parameters but we have %d standard deviations."
                            .formatted(getDimensions(), stdDevs.size()));
        }
        for (double stdDev : stdDevs) {
            if (stdDev <= 0) {
                throw new IllegalArgumentException(
                        "All standard deviations must be positive " +
                                "(%f <= 0)".formatted(stdDev));
            }
        }
        this.means = means;
        this.stdDevs = stdDevs;
    }

    /** Generates normally distributed data.
     *
     */
    @Override
    void generateData() {
        List<List<Double>> xData = Stream
                .iterate(0, i -> i < getDimensions(), i -> i + 1)
                .map(i -> Stream
                        .generate(() -> getRandom().nextGaussian(means.get(i), stdDevs.get(i)))
                        .limit(getNumDatapoints())
                        .toList())
                .toList();
        addData(xData);
    }
}
