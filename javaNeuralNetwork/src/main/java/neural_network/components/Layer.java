package neural_network.components;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/** Class to represent one {@code Layer} of a network.
 * @version 1.0.0
 * @since 1.0.0
 */
public class Layer {

    private final int id;
    private final int numNeurons;
    private final List<Neuron> neurons = new ArrayList<>();

    /** Constructor method.
     *
     * @param id The id of the {@code Layer}.
     * @param numNeurons The number of {@code Neurons} in the {@code Layer}.
     */
    Layer(int id, int numNeurons) {
        this.id = id;
        this.numNeurons = numNeurons;
        this.neurons.addAll(Stream
                .iterate(0, j -> j < numNeurons, j -> j + 1)
                .map(j -> new Neuron(id, j))
                .toList());
    }

    /** Getter method for {@code id}.
     *
     * @return The {@code id} of the {@code Layer}.
     */
    int getId() {
        return id;
    }

    /** Getter method for {@code neurons}
     *
     * @return A {@code List} of {@code Neurons} in the {@code Layer}.
     */
    List<Neuron> getNeurons() {
        return List.copyOf(neurons);
    }

    /** The size of the {@code Layer}.
     *
     * @return The number of {@code Neurons} in the {@code Layer}.
     */
    int size() {
        return numNeurons;
    }

    /** String method.
     *
     * @return {@code String} representation of the {@code Layer}.
     */
    @Override
    public String toString() {
        return "Layer %d".formatted(id);
    }
}
