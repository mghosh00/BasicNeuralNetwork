package neural_network.util;

import java.util.ArrayList;
import java.util.List;

/** Enum to represent the column headers of a dataframe for datapoints.
 *
 *
 */
public enum Header {
    /** First x coordinate.
     *
     */
    X_1,
    /** Second x coordinate.
     *
     */
    X_2,
    /** Third x coordinate.
     *
     */
    X_3,
    /** Fourth x coordinate.
     *
     */
    X_4,
    /** Fifth x coordinate.
     *
     */
    X_5,
    /** Sixth x coordinate.
     *
     */
    X_6,
    /** Seventh x coordinate.
     *
     */
    X_7,
    /** Eighth x coordinate.
     *
     */
    X_8,
    /** Ninth x coordinate.
     *
     */
    X_9,
    /** True class/value of the datapoint.
     *
     */
    Y,
    /** Predicted class/value of the datapoint.
     *
     */
    Y_HAT;

    private static int dimensions = 0;

    /** Sets up the dimensions (number of x coordinates) for the datapoints.
     *
     * @param dimensions The number of x coordinates for the datapoints.
     * @throws IllegalArgumentException If {@code dimensions} is not between 0 and 9.
     */
    public static void setDimensions(int dimensions) {
        if (0 <= dimensions && dimensions <= 9) {
            Header.dimensions = dimensions;
        } else {
            throw new IllegalArgumentException("Invalid dimensions passed: %d, "
                    .formatted(dimensions) + "must be between 0 and 9.");
        }
    }

    /** Gets a {@code Header} constant from a given name (upper or lower case).
     *
     * @param name The name of the {@code Header}.
     * @return The constant associated with the {@code name}.
     */
    public static Header getHeader(String name) {
        return Header.valueOf(name.toUpperCase());
    }

    /** Retrieves a list of {@code Headers} including the {@code X} coordinate headers
     * and the {@code Y} header (the true class/value of the datapoint).
     *
     * @return The above described list for the dimensions we have.
     */
    public static List<Header> getInitialHeaders() {
        if (dimensions > 0) {
            List<Header> headers = new ArrayList<>(
                    List.of(Header.values()).subList(0, dimensions));
            headers.add(Y);
            return headers;
        } else {
            throw new IllegalStateException("dimensions of Header not yet initialised");
        }
    }

    /** Retrieves a list of {@code Headers} including the {@code X} coordinate headers,
     * the {@code Y} header (the true class/value of the datapoint) and the {@code Y_HAT}
     * header (the predicted class/value of the datapoint).
     *
     * @return The above described list for the dimensions we have.
     */
    public static List<Header> getAllHeaders() {
        if (dimensions > 0) {
            List<Header> headers = getInitialHeaders();
            headers.add(Y_HAT);
            return headers;
        } else {
            throw new IllegalStateException("dimensions of Header not yet initialised");
        }
    }

    /** Getter method for {@code dimensions}.
     *
     * @return The number of dimensions.
     */
    public static int getDimensions() {
        return dimensions;
    }

    /** String representation.
     *
     * @return The lower case version of the enum.
     */
    @Override
    public String toString() {
        return this.name().toLowerCase();
    }
}
