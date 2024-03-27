package neural_network.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class NeuronTest {
    private Neuron neuron;

    @BeforeEach
    void setUp() {
        this.neuron = new Neuron(3, 4);
    }

    @Test
    void construct() {
        assertIterableEquals(new ArrayList<>(List.of(3, 4)), neuron.getId());
        assertEquals(0.0, neuron.getBias());
        assertEquals(Double.NaN, neuron.getValue());
        assertIterableEquals(new ArrayList<>(), neuron.getBiasGradients());
    }

    @Test
    void addBiasGradient() {
        neuron.addBiasGradient(0.1);
        neuron.addBiasGradient(0.2);
        assertIterableEquals(new ArrayList<>(List.of(0.1, 0.2)),
                             neuron.getBiasGradients());
    }

    @Test
    void clearBiasGradients() {
        neuron.addBiasGradient(0.4);
        assertIterableEquals(new ArrayList<>(List.of(0.4)),
                             neuron.getBiasGradients());
        neuron.clearBiasGradients();
        assertIterableEquals(new ArrayList<>(), neuron.getBiasGradients());
    }

    @Test
    void setBias() {
        neuron.setBias(0.5);
        assertEquals(0.5, neuron.getBias());
    }

    @Test
    void setValue() {
        neuron.setValue(0.6);
        assertEquals(0.6, neuron.getValue());
    }

    @Test
    void string() {
        assertEquals("Neuron [3, 4]", neuron.toString());
    }
}
