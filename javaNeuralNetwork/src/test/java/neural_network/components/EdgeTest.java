package neural_network.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class EdgeTest {
    private Neuron leftNeuron;
    private Neuron rightNeuron;
    private Edge edge;

    @BeforeEach
    void setUp() {
        Random mockRandom = mock(Random.class);
        when(mockRandom.nextDouble(-1, 1)).thenReturn(0.1);
        Edge.setRandom(mockRandom);
        this.leftNeuron = new Neuron(4, 3);
        this.rightNeuron = new Neuron(5, 2);
        this.edge = new Edge(leftNeuron, rightNeuron);
    }

    @Test
    void constructErroneous() {
        Exception exception = assertThrows(RuntimeException.class,
                () -> new Edge(rightNeuron, leftNeuron));
        assertEquals("Edge must connect adjacent layers (left: 5, right: 4)",
                exception.getMessage());
    }

    @Test
    void construct() {
        assertEquals(leftNeuron, edge.getLeftNeuron());
        assertEquals(rightNeuron, edge.getRightNeuron());
        assertIterableEquals(new ArrayList<>(List.of(4, 3, 2)), edge.getId());
        assertEquals(0.1, edge.getWeight());
        assertIterableEquals(new ArrayList<>(), edge.getLossGradients());
        assertEquals(0.0, edge.getDelta());
        assertEquals(0.0, edge.getVelocity());
    }

    @Test
    void addLossGradient() {
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.2);
        assertIterableEquals(new ArrayList<>(List.of(0.1, 0.2)),
                edge.getLossGradients());
    }

    @Test
    void clearLossGradients() {
        edge.addLossGradient(0.4);
        assertIterableEquals(new ArrayList<>(List.of(0.4)),
                edge.getLossGradients());
        edge.clearLossGradients();
        assertIterableEquals(new ArrayList<>(), edge.getLossGradients());
    }

    @Test
    void setWeight() {
        edge.setWeight(-0.5);
        assertEquals(-0.5, edge.getWeight());
    }

    @Test
    void setDelta() {
        edge.setDelta(0.9);
        assertEquals(0.9, edge.getDelta());
    }

    @Test
    void setVelocity() {
        edge.setVelocity(0.3);
        assertEquals(0.3, edge.getVelocity());
    }

    @Test
    void string() {
        assertEquals("Edge [4, 3, 2]", edge.toString());
    }
}
