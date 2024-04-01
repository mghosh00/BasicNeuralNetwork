package neural_network.data_generators;

import neural_network.util.Header;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.stream.Stream;

/** Class to randomly generate datapoints and categorise them according to
 * a given rule, or provide an output value if we are regressing,
 * with data being generated via a normal distribution.
 *
 * @param <T> The return type of the {@code function} (could be any type
 *           for a classification problem).
 */
public abstract class DataGenerator<T> {
    
    private final Method function;
    private final int dimensions;
    private final int numDatapoints;
    private final List<Header> headers;
    private final NavigableMap<Header, List<String>> df = new TreeMap<>();
    private final List<List<Double>> xData = new ArrayList<>();
    private static Random random = new Random();
    private CSVPrinter printer = null;

    /** Constructor method for {@code DataGenerator}. Importantly, the {@code df} is
     * set up here, which consists of a map of column {@code Headers} as keys and lists
     * of doubles as values. The length of each of these lists will be equal to the
     * number of datapoints, as each {@code Header} represents a single coordinate.
     *
     * @param function A rule which takes a certain number of coordinates and returns a
     *                 value representing the class or function output of the datapoint.
     *                 This must be a static method from the underlying class.
     * @param numDatapoints The number of datapoints to be generated.
     * @throws IllegalArgumentException If the parameter types of {@code function}
     * are not {@code doubles}, if the number of coordinates are invalid or if
     * {@code numDatapoints} is not greater than {@code 0}.
     */
    public DataGenerator(Method function, int numDatapoints) {
        // Checks on function parameter types; return type must match T
        for (Class<?> type : function.getParameterTypes()) {
            if (! (type == double.class)) {
                throw new IllegalArgumentException("All parameters of function" +
                        "must be doubles, {%s} illegal".formatted(type));
            }
        }
        if (! (Modifier.isStatic(function.getModifiers()))) {
            throw new IllegalArgumentException("function must be a static method");
        }
        this.function = function;
        this.dimensions = function.getParameterCount();
        if (dimensions < 1 || dimensions > 9) {
            throw new IllegalArgumentException("function must have 1 - 9 coordinates, " +
                    "numCoordinates = %d".formatted(dimensions));
        }
        if (numDatapoints < 1) {
            throw new IllegalArgumentException("Must have at least 1 datapoint, " +
                    "numDatapoints = %d".formatted(numDatapoints));
        }
        this.numDatapoints = numDatapoints;
        // Below, set up all the headers in the df using the static methods below,
        // making sure to set the correct number of dimensions first
        Header.setDimensions(dimensions);
        this.headers = Header.getInitialHeaders();
        headers.forEach(header -> this.df.put(header, new ArrayList<>()));
    }

    /** To generate the data using a specific probability distribution from
     * a subclass.
     *
     */
    abstract void generateData();

    /** Writes to the {@code df} with generated data and the specific classes
     * or regression values (depending on whether this is for a classification
     * or regression problem).
     *
     * @return {@code df} with the newly generated data.
     * @throws InvocationTargetException If the underlying {@code function} throws an exception.
     * @throws IllegalAccessException If the underlying {@code function} is not public.
     */
    @SuppressWarnings("unchecked")
    public NavigableMap<Header, List<String>> call() throws InvocationTargetException, IllegalAccessException {
        // Generates data (using a subclass)
        generateData();

        // Update df with xData (list of lists, size (dimensions x numDatapoints))
        // We also convert all values to strings in preparation for sending this
        // to a .csv file
        for (int i = 0; i < dimensions; i ++) {
            df.get(headers.get(i)).addAll(xData.get(i).stream()
                    .map(Object::toString)
                    .toList());
        }

        for (int j = 0; j < numDatapoints; j ++) {
            // First convert to an Object array
            int finalJ = j;
            Object[] x = xData.stream()
                    .map(list -> list.get(finalJ))
                    .toArray();

            // The below will evaluate the function for one datapoint x_j
            // using all its coordinates as inputs. We use null on the
            // invoke method, as we know that function is a static method
            T value = (T) function.invoke(null, x);
            df.get(Header.Y).add(value.toString());
        }

        return df;
    }

    /** Writes the generated data to a .csv file.
     *
     * @param title The title for the .csv file.
     * @param directory The directory for the file.
     * @throws RuntimeException If the directory does not exist.
     */
    public void writeToCsv(String title, String directory) {
        String path = !directory.isEmpty() ? "%s/%s.csv".formatted(directory, title)
                : "%s.csv".formatted(title);
        try {
            if (printer == null) {
                printer = new CSVPrinter(new FileWriter(path), CSVFormat.DEFAULT);
            }
            // Write a row for the headers
            printer.printRecord(headers);
            for (int j = 0; j < numDatapoints; j ++) {
                int finalJ = j;

                // The list below contains all values for one datapoint
                List<String> record = Stream
                        .iterate(0, i -> i < dimensions + 1, i -> i + 1)
                        .map(i -> df.get(headers.get(i)).get(finalJ))
                        .toList();

                // Write a row for this datapoint
                printer.printRecord(record);
            }
            printer.flush();
        } catch (IOException e) {
            throw new RuntimeException("Path %s does not exist or is otherwise invalid."
                    .formatted(path));
        }
    }

    /** Writes the generated data to a .csv file.
     *
     * @param title The title for the .csv file.
     */
    public void writeToCsv(String title) {
        writeToCsv(title, "");
    }

    /** Getter method for dimensions - only needed for package access.
     *
     * @return The number of dimensions.
     */
    int getDimensions() {
        return dimensions;
    }

    /** Getter method for numDatapoints - only needed for package access.
     *
     * @return The number of datapoints.
     */
    int getNumDatapoints() {
        return numDatapoints;
    }

    /** Adder method for xData - only needed for package access.
     *
     * @param data The generated data from a subclass.
     */
    void addData(List<List<Double>> data) {
        xData.addAll(data);
    }

    /** Getter method for {@code random}.
     *
     * @return The {@code random}.
     */
    static Random getRandom() {
        return random;
    }

    /** Setter method for {@code random}.
     *
     * @param random The new {@code random}. This can be used for mocking
     *               purposes.
     */
    static void setRandom(Random random) {
        DataGenerator.random = random;
    }

    /** Setter method for {@code printer}.
     *
     * @param printer The new {@code printer}. This can be used for mocking
     *                purposes.
     */
    void setPrinter(CSVPrinter printer) {
        this.printer = printer;
    }

    /** Getter method for {@code printer}.
     *
     * @return The {@code printer}. This is used for testing purposes.
     */
    CSVPrinter getPrinter() {
        return printer;
    }
}
