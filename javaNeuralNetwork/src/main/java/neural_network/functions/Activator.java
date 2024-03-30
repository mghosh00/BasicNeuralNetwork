package neural_network.functions;

/** Interface for an activating function in a neural network.
 *
 * @param <T> The type of the input parameter to the {@code call} method. This
 *           will typically be either a {@code Double} or {@code List} of
 *           {@code Doubles} depending on the purpose of the {@code Activator}.
 */
public interface Activator<T> {

    /** Calling of the activating function.
     *
     * @param x The input value (typically {@code Double} or {@code List} of
     *          {@code Doubles}).
     * @return The output value.
     */
    double call(T x);

    /** The gradient of the activating function to be used in back propagation.
     *
     * @param x The input value (typically {@code Double} or {@code List} of
     *          {@code Doubles}).
     * @return The gradient of the function at {@code x}.
     */
    T gradient(T x);
}
