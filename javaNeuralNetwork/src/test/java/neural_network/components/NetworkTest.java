package neural_network.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.*;

public class NetworkTest {

    private static final List<Double> preActivatedValues = new ArrayList<>(List.of(
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0));
    private Network defaultNetwork;
    private Network network;
    private Network minimalNetwork;
    private Network regressionNetwork;

    @BeforeEach
    void setUp() {
        // Edge weights always initialised to 0.5, He weights are 0.2
        Random mockRandom = mock(Random.class);
        when(mockRandom.nextDouble(-1, 1)).thenReturn(0.5);
        when(mockRandom.nextGaussian(anyDouble(), anyDouble())).thenReturn(0.2);
        Edge.setRandom(mockRandom);
        Network.setRandom(mockRandom);
        defaultNetwork = new Network(3, 2, new ArrayList<>(
                List.of(4, 2)));
        network = new Network(2, 3, new ArrayList<>(
                List.of(1, 4, 2)), 3, 0.5, 0.005,
                false, true, 0.8, true);

        // Uses the medium level constructor
        minimalNetwork = new Network(1, 0, new ArrayList<>(),
                2, 0.01, 0.01);
        regressionNetwork = new Network(2, 3, new ArrayList<>(
                List.of(1, 4, 2)), 3, 0.5, 0.005,
                true, true, 0.8, true);
    }

    @Test
    void constructErroneous() {
        Exception exception = assertThrows(IllegalArgumentException.class,
                () -> new Network(1, 2, new ArrayList<>(
                        List.of(3, 4, 5))));
        assertEquals("neuronCounts (3) must have a length equal to " +
                "numHiddenLayers (2)", exception.getMessage());
    }

    @Test
    void constructDefault() {
        assertFalse(defaultNetwork.isRegressor());
        List<Integer> expectedNeuronCounts = List.of(3, 4, 2, 2);
        assertIterableEquals(expectedNeuronCounts, defaultNetwork.getNeuronCounts());
    }

    @Test
    void constructLayers() {
        List<Integer> expectedNeuronCounts = List.of(2, 1, 4, 2, 3);
        assertIterableEquals(expectedNeuronCounts, network.getNeuronCounts());

        // Checking layers for main network
        List<Layer> layers = network.getLayers();
        for (int i = 0; i < 5; i ++) {
            Layer layer = layers.get(i);
            assertEquals(i, layer.getId());
            assertEquals(expectedNeuronCounts.get(i), layer.size());
        }

        // Checking layers for minimal network
        expectedNeuronCounts = List.of(1, 2);
        assertIterableEquals(expectedNeuronCounts, minimalNetwork.getNeuronCounts());
        List<Layer> minimalLayers = minimalNetwork.getLayers();
        for (int i = 0; i < 2; i ++) {
            Layer layer = minimalLayers.get(i);
            assertEquals(i, layer.getId());
            assertEquals(expectedNeuronCounts.get(i), layer.size());
        }
    }

    @Test
    void constructEdges() {
        // Checking edges
        List<List<List<Edge>>> edges = network.getEdges();
        assertEquals(4, edges.size());
        for (int i = 0; i < 4; i ++) {
            List<List<Edge>> leftLayer = edges.get(i);
            for (int j = 0; j < leftLayer.size(); j ++) {
                List<Edge> rightNeuron = leftLayer.get(j);
                for (int k = 0; k < rightNeuron.size(); k ++) {
                    Edge edge = edges.get(i).get(j).get(k);
                    assertIterableEquals(List.of(i, k, j), edge.getId());
                    // Check for He weights
                    assertEquals(0.2, edge.getWeight());
                }
            }
        }

        // Check non-He weights for default network
        edges = defaultNetwork.getEdges();
        for (int i = 0; i < 3; i ++) {
            List<List<Edge>> leftLayer = edges.get(i);
            for (int j = 0; j < leftLayer.size(); j ++) {
                List<Edge> rightNeuron = leftLayer.get(j);
                for (int k = 0; k < rightNeuron.size(); k ++) {
                    Edge edge = edges.get(i).get(j).get(k);
                    assertIterableEquals(List.of(i, k, j), edge.getId());
                    // Check for He weights
                    assertEquals(0.5, edge.getWeight());
                }
            }
        }
    }

    @Test
    void constructRegressionNetwork() {
        assertTrue(regressionNetwork.isRegressor());

        // Check that the number of classes is 1 even though 3 was passed to the
        // constructor
        List<Integer> expectedNeuronCounts = List.of(2, 1, 4, 2, 1);
        assertIterableEquals(expectedNeuronCounts, regressionNetwork.getNeuronCounts());
    }

    @Test
    void forwardPassOneDatapointErroneous() {
        List<Double> erroneousData = new ArrayList<>(List.of(1.0, 2.0, 3.0));
        Exception exception = assertThrows(IllegalArgumentException.class,
                () -> network.forwardPassOneDatapoint(erroneousData));
        assertEquals("Number of features must match the number of neurons " +
                        "in the input layer (3 != 2)", exception.getMessage());
    }

    @Test
    void forwardPassOneDatapoint() {
        Network spyNetwork = spy(network);
        doAnswer(new Answer<>() {
            int j = -1;

            @Override
            public Object answer(InvocationOnMock invocationOnMock) {
                j++;
                return preActivatedValues.get(j);
            }
        }).when(spyNetwork).calculatePreActivatedValue(any(Layer.class), any(Neuron.class));
        doReturn(new ArrayList<>(List.of(0.2, 0.4, 0.4)))
                .when(spyNetwork).activateOutputLayer(any());
        List<Double> x = new ArrayList<>(List.of(2.0, 3.0));
        List<Double> softmaxVector = spyNetwork.forwardPassOneDatapoint(x);

        // Check input layer has correct values
        Layer inputLayer = spyNetwork.getLayers().get(0);
        for (int i = 0; i < inputLayer.size(); i ++) {
            Neuron inputNeuron = inputLayer.getNeurons().get(i);
            assertEquals(x.get(i), inputNeuron.getValue());
        }

        // All layers other than the output layer
        List<Layer> mainLayers = spyNetwork.getLayers().subList(0, 4);
        int index = 0;
        for (Layer leftLayer : mainLayers.subList(0, 3)) {
            Layer rightLayer = mainLayers.get(leftLayer.getId() + 1);
            for (Neuron rightNeuron : rightLayer.getNeurons()) {
                verify(spyNetwork).calculatePreActivatedValue(leftLayer, rightNeuron);
                assertEquals(preActivatedValues.get(index), rightNeuron.getValue());
                index ++;
            }
        }

        verify(spyNetwork, times(10))
                .calculatePreActivatedValue(any(Layer.class), any(Neuron.class));

        assertIterableEquals(List.of(0.2, 0.4, 0.4), softmaxVector);
    }

    @Test
    void forwardPassOneDatapointRegression() {
        Network spyNetwork = spy(regressionNetwork);
        doAnswer(new Answer<>() {
            int j = -1;

            @Override
            public Object answer(InvocationOnMock invocationOnMock) {
                j++;
                return preActivatedValues.get(j);
            }
        }).when(spyNetwork).calculatePreActivatedValue(any(Layer.class), any(Neuron.class));
        doReturn(new ArrayList<>(List.of(0.2, 0.4, 0.4)))
                .when(spyNetwork).activateOutputLayer(any());
        List<Double> x = new ArrayList<>(List.of(2.0, 3.0));
        List<Double> predList = spyNetwork.forwardPassOneDatapoint(x);

        // Check input layer has correct values
        Layer inputLayer = spyNetwork.getLayers().get(0);
        for (int i = 0; i < inputLayer.size(); i ++) {
            Neuron inputNeuron = inputLayer.getNeurons().get(i);
            assertEquals(x.get(i), inputNeuron.getValue());
        }

        // All layers other than the output layer
        List<Layer> mainLayers = spyNetwork.getLayers().subList(0, 4);
        int index = 0;
        for (Layer leftLayer : mainLayers.subList(0, 3)) {
            Layer rightLayer = mainLayers.get(leftLayer.getId() + 1);
            for (Neuron rightNeuron : rightLayer.getNeurons()) {
                verify(spyNetwork).calculatePreActivatedValue(leftLayer, rightNeuron);
                assertEquals(preActivatedValues.get(index), rightNeuron.getValue());
                index ++;
            }
        }

        verify(spyNetwork, times(8))
                .calculatePreActivatedValue(any(Layer.class), any(Neuron.class));

        assertIterableEquals(List.of(0.8), predList);
    }

    @Test
    void calculatePreActivatedValue() {
        // Here, we wish to control all values involved so that we can
        // track the calculation

        // We choose the leftLayer to have 4 neurons, and the rightNeuron
        // to be the second one down in its layer
        Layer leftLayer = network.getLayers().get(2);
        List<Neuron> leftNeurons = leftLayer.getNeurons();
        assertEquals(4, leftLayer.size());
        Neuron rightNeuron = network.getLayers().get(3).getNeurons().get(1);

        // This list contains 4 edges all connected to rightNeuron from
        // the leftLayer
        List<Edge> edges = network.getEdges().get(2).get(1);
        assertEquals(4, edges.size());

        // Setting up values
        for (int i = 0; i < 4; i ++) {
            Neuron leftNeuron = leftNeurons.get(i);
            // values = 2, 3, 4, 5
            leftNeuron.setValue(i + 2);
        }
        for (int j = 0; j < 4; j ++) {
            Edge edge = edges.get(j);
            // weights = 2, 1, 0, -1
            edge.setWeight(2 - j);
        }
        rightNeuron.setBias(2);

        // This method will apply the transfer function followed by a Leaky ReLU,
        // with leak = 0.5
        double z1 = network.calculatePreActivatedValue(leftLayer, rightNeuron);
        assertEquals(4.0, z1);

        // Now try with different weights, so that the transfer is negative
        for (int j = 0; j < 4; j ++) {
            Edge edge = edges.get(j);
            // weights = 1, 0, -1, -2
            edge.setWeight(1 - j);
        }
        double z2 = network.calculatePreActivatedValue(leftLayer, rightNeuron);
        assertEquals(-10.0, z2);
    }

    @Test
    void activateOutputLayer() {
        // Now control all values from the output layer
        List<Double> zList = new ArrayList<>(List.of(-2.0, 0.0, 2.0));
        List<Double> softmaxVector = network.activateOutputLayer(zList);

        List<Double> predictedVector = new ArrayList<>(
                List.of(0.01587624, 0.11731043, 0.86681333));
        List<Neuron> softmaxNeurons = network.getLayers().get(4).getNeurons();
        for (int i = 0; i < 3; i ++) {
            assertEquals(predictedVector.get(i), softmaxVector.get(i), .00000001);
            assertEquals(predictedVector.get(i), softmaxNeurons.get(i).getValue(),
                    .00000001);
        }
    }

    @Test
    void storeGradientOfLoss() {
        // index = 4 is output layer
        // edge location: leftLayer = 3, rightNeuron = 1, leftNeuron = 0
        Edge edge = network.getEdges().get(3).get(1).get(0);

        // We have 3 classes, so target = 0, 1 or 2
        int target = 0;
        boolean first = true;

        // Set up values
        edge.getLeftNeuron().setValue(0.2);
        Neuron rightNeuron = edge.getRightNeuron();
        rightNeuron.setValue(-0.5);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.3);
        rightNeuron.addBiasGradient(-0.3);
        rightNeuron.addBiasGradient(0.1);
        rightNeuron.addBiasGradient(0.4);

        network.storeGradientOfLoss(edge, target, first);
        assertEquals(-0.5, edge.getDelta());
        assertIterableEquals(List.of(0.2, 0.1, 0.3, -0.1), edge.getLossGradients());
        assertIterableEquals(List.of(-0.3, 0.1, 0.4, -0.5), rightNeuron.getBiasGradients());

        // Now try for a new edge
        edge = network.getEdges().get(3).get(0).get(1);
        target = 0;
        first = false;

        edge.getLeftNeuron().setValue(0.2);
        rightNeuron = edge.getRightNeuron();
        rightNeuron.setValue(-0.5);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.3);
        rightNeuron.addBiasGradient(-0.3);
        rightNeuron.addBiasGradient(0.1);
        rightNeuron.addBiasGradient(0.4);

        network.storeGradientOfLoss(edge, target, first);
        assertEquals(-1.5, edge.getDelta());
        List<Double> expected = List.of(0.2, 0.1, 0.3, -0.3);
        for (int i = 0; i < 4; i ++) {
            assertEquals(edge.getLossGradients().get(i), expected.get(i),
                    .00000001);
        }
        assertIterableEquals(List.of(-0.3, 0.1, 0.4), rightNeuron.getBiasGradients());
    }

    @Test
    void storeGradientOfLossOuterLayerRegression() {
        // index = 4 is output layer
        // edge location: leftLayer = 3, rightNeuron = 0, leftNeuron = 0
        Edge edge = regressionNetwork.getEdges().get(3).get(0).get(0);

        // Choose ground truth value
        double target = 8.8;
        boolean first = true;

        // Set up values
        edge.getLeftNeuron().setValue(0.2);
        Neuron rightNeuron = edge.getRightNeuron();
        rightNeuron.setValue(8.0);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.3);
        rightNeuron.addBiasGradient(-0.3);
        rightNeuron.addBiasGradient(0.1);
        rightNeuron.addBiasGradient(0.4);

        regressionNetwork.storeGradientOfLoss(edge, target, first);
        assertEquals(-1.6, edge.getDelta(), .00000001);
        List<Double> expectedLossGradients = List.of(0.2, 0.1, 0.3, -0.32);
        List<Double> expectedBiasGradients = List.of(-0.3, 0.1, 0.4, -1.6);
        for (int i = 0; i < 4; i ++) {
            assertEquals(edge.getLossGradients().get(i),
                    expectedLossGradients.get(i), .00000001);
            assertEquals(rightNeuron.getBiasGradients().get(i),
                    expectedBiasGradients.get(i),.00000001);
        }

        // Now try for a new edge
        edge = regressionNetwork.getEdges().get(3).get(0).get(1);
        first = false;

        edge.getLeftNeuron().setValue(0.2);
        rightNeuron = edge.getRightNeuron();
        rightNeuron.setValue(8.0);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.3);
        rightNeuron.clearBiasGradients();
        rightNeuron.addBiasGradient(-0.3);
        rightNeuron.addBiasGradient(0.1);
        rightNeuron.addBiasGradient(0.4);

        regressionNetwork.storeGradientOfLoss(edge, target, first);
        assertEquals(-1.6, edge.getDelta(), .00000001);
        List<Double> expected = List.of(0.2, 0.1, 0.3, -0.32);
        for (int i = 0; i < 4; i ++) {
            assertEquals(edge.getLossGradients().get(i), expected.get(i),
                    .00000001);
        }
        assertIterableEquals(List.of(-0.3, 0.1, 0.4), rightNeuron.getBiasGradients());
    }

    @Test
    void storeGradientOfLossHiddenLayer() {
        // edge location: leftLayer = 2, rightNeuron = 1, leftNeuron = 0
        Edge edge = network.getEdges().get(2).get(1).get(0);

        edge.getLeftNeuron().setValue(0.2);
        Neuron rightNeuron = edge.getRightNeuron();
        rightNeuron.setValue(-0.5);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.3);
        rightNeuron.addBiasGradient(-0.3);
        rightNeuron.addBiasGradient(0.1);
        rightNeuron.addBiasGradient(0.4);

        // There are 3 edges connecting the rightNeuron to the next layer
        // along, and we must take these values into account also
        List<Edge> newEdges = Stream
                .iterate(0, j -> j < 3, j -> j + 1)
                .map(j -> network.getEdges().get(3).get(j).get(1))
                .toList();
        for (int j = 0; j < 3; j ++) {
            Edge newEdge = newEdges.get(j);
            newEdge.setWeight(j + 1);
            newEdge.setDelta(-j);
        }

        int target = 0;
        boolean first = true;
        network.storeGradientOfLoss(edge, target, first);

        // 1 * 0 + 2 * (-1) + 3 * (-2)
        double factor = -8.0;
        double reluGrad = 0.5;
        assertEquals(factor * reluGrad, edge.getDelta());
        List<Double> expected = List.of(0.2, 0.1, 0.3, -0.8);
        for (int i = 0; i < 4; i ++) {
            assertEquals(edge.getLossGradients().get(i), expected.get(i),
                    .00000001);
        }
        List<Double> expectedBiasGradients = List.of(-0.3, 0.1, 0.4, -4.0);
        for (int i = 0; i < 4; i ++) {
            assertEquals(rightNeuron.getBiasGradients().get(i),
                    expectedBiasGradients.get(i),
                    .00000001);
        }
    }

    @Test
    void storeGradientOfLossHiddenLayerNotFirst() {
        // edge location: leftLayer = 2, rightNeuron = 1, leftNeuron = 0
        Edge edge = network.getEdges().get(2).get(1).get(0);

        edge.getLeftNeuron().setValue(0.2);
        Neuron rightNeuron = edge.getRightNeuron();
        rightNeuron.setValue(-0.5);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.3);
        rightNeuron.addBiasGradient(-0.3);
        rightNeuron.addBiasGradient(0.1);
        rightNeuron.addBiasGradient(0.4);

        // There are 3 edges connecting the rightNeuron to the next layer
        // along, and we must take these values into account also
        List<Edge> newEdges = Stream
                .iterate(0, j -> j < 3, j -> j + 1)
                .map(j -> network.getEdges().get(3).get(j).get(1))
                .toList();
        for (int j = 0; j < 3; j ++) {
            Edge newEdge = newEdges.get(j);
            newEdge.setWeight(j + 1);
            newEdge.setDelta(-j);
        }

        int target = 0;
        boolean first = false;
        network.storeGradientOfLoss(edge, target, first);

        // 1 * 0 + 2 * (-1) + 3 * (-2)
        double factor = -8.0;
        double reluGrad = 0.5;
        assertEquals(factor * reluGrad, edge.getDelta());
        List<Double> expected = List.of(0.2, 0.1, 0.3, -0.8);
        for (int i = 0; i < 4; i ++) {
            assertEquals(edge.getLossGradients().get(i), expected.get(i),
                    .00000001);
        }
        List<Double> expectedBiasGradients = List.of(-0.3, 0.1, 0.4);
        for (int i = 0; i < 3; i ++) {
            assertEquals(rightNeuron.getBiasGradients().get(i),
                    expectedBiasGradients.get(i),
                    .00000001);
        }
    }

    @Test
    void backPropagateWeightNoMomentum() {
        // The default network has adaptive = false
        Edge edge = defaultNetwork.getEdges().get(1).get(1).get(3);

        // Current weight
        edge.setWeight(1.0);

        // The loss gradients from the batch
        edge.addLossGradient(-0.1);
        edge.addLossGradient(0.0);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.3);
        defaultNetwork.backPropagateWeight(edge);

        // Learning rate is 0.01
        assertIterableEquals(List.of(), edge.getLossGradients());
        assertEquals(1 - 0.01 * 0.1, edge.getWeight());
    }

    @Test
    void backPropagateWeightWithNoMomentum() {
        // The network has adaptive = true
        Edge edge = network.getEdges().get(1).get(1).get(0);

        // Current weight and velocity for edge
        edge.setWeight(1.0);
        edge.setVelocity(-2.0);

        // The loss gradients picked up by the batch
        edge.addLossGradient(-0.1);
        edge.addLossGradient(0.0);
        edge.addLossGradient(0.1);
        edge.addLossGradient(0.2);
        edge.addLossGradient(0.3);

        network.backPropagateWeight(edge);

        // learningRate is 0.005 and gamma is 0.8
        assertIterableEquals(List.of(), edge.getLossGradients());
        assertEquals(1 + 2 * 0.8 - 0.005 * 0.1, edge.getWeight());
        assertEquals(0.005 * 0.1 - 2 * 0.8, edge.getVelocity());
    }

    @Test
    void backPropagateWeights() {
        Network spyNetwork = spy(network);
        doNothing().when(spyNetwork).backPropagateWeight(any(Edge.class));
        spyNetwork.backPropagateWeights();
        for (List<List<Edge>> list1 : spyNetwork.getEdges()) {
            for (List<Edge> list2 : list1) {
                for (Edge edge : list2) {
                    verify(spyNetwork, times(1))
                            .backPropagateWeight(edge);
                }
            }
        }
    }

    @Test
    void backPropagateBias() {
        Neuron neuron = network.getLayers().get(3).getNeurons().get(0);

        // Current bias for neuron
        neuron.setBias(2.0);

        // The bias gradients picked up by the batch
        neuron.addBiasGradient(0.1);
        neuron.addBiasGradient(0.2);
        neuron.addBiasGradient(0.3);
        neuron.addBiasGradient(0.4);
        network.backPropagateBias(neuron);

        // Check they have been reset
        assertIterableEquals(List.of(), neuron.getBiasGradients());

        assertEquals(neuron.getBias(), 2 - 0.005 * 0.25);
    }

    @Test
    void backPropagateBiases() {
        Network spyNetwork = spy(network);
        doNothing().when(spyNetwork).backPropagateBias(any(Neuron.class));
        spyNetwork.backPropagateBiases();
        for (Layer layer : spyNetwork.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                if (neuron.getId().get(0) == 0) {
                    verify(spyNetwork, times(0))
                            .backPropagateBias(neuron);
                } else {
                    verify(spyNetwork, times(1))
                            .backPropagateBias(neuron);
                }
            }
        }
    }
}
