package neural_network.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

public class LayerTest {

    private Layer layer;

    @BeforeEach
    void setUp() {
        this.layer = new Layer(2, 6);
    }

    @Test
    void construct() {
        assertEquals(2, layer.getId());
        assertEquals(6, layer.size());
        List<Neuron> neurons = layer.getNeurons();
        for (int j = 0; j < 6; j ++) {
            Neuron neuron = neurons.get(j);
            assertIterableEquals(List.of(2, j), neuron.getId());
        }
    }

    @Test
    void string() {
        assertEquals("Layer 2", layer.toString());
    }
}
