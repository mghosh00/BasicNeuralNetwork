package neural_network.learning;

import neural_network.components.Edge;
import neural_network.components.Network;
import neural_network.functions.CrossEntropyLoss;
import neural_network.functions.MSELoss;
import neural_network.util.Header;
import neural_network.util.Partitioner;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class TrainerTest extends LearnerTest {

    private final List<List<Integer>> partitions = List.of(
            List.of(6, 2), List.of(1, 8), List.of(0, 3), List.of(4, 7), List.of(9, 5));
    private final List<List<Integer>> weightedPartitions = List.of(
            List.of(5, 2), List.of(2, 8), List.of(4, 1), List.of(8, 6), List.of(9, 3));
    private final List<Double> batchLosses = List.of(
            0.1, 0.2, 0.3, 0.2, 0.5,
            0.6, 0.2, 0.3, 0.1, 0.3,
            0.2, 0.1, 0.0, 0.0, 0.2,
            0.1, 0.1, 0.1, 0.1, 0.1,
            0.0, 0.1, 0.0, 0.1, 0.0);
    private final List<Double> validationLosses = List.of(0.9, 0.7, 0.5, 0.3, 0.1);
    private final NavigableMap<Header, List<String>> validationDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("4", "2", "-2", "-9"),
            Header.X_2, List.of("1", "5", "-4", "2"),
            Header.X_3, List.of("3", "-4", "1", "4"),
            Header.Y, List.of("r", "r", "l", "l")
    ));
    private final NavigableMap<Header, List<String>> trainingDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4", "2", "-9", "2"),
            Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2", "3", "-3", "3"),
            Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5", "1", "2", "-4"),
            Header.Y, List.of("r", "r", "r", "l", "l", "r", "l", "r", "l", "r")
    ));
    private final NavigableMap<Header, List<String>> categoricalDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4", "2", "-9", "2"),
            Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2", "3", "-3", "3"),
            Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5", "1", "2", "-4"),
            Header.Y, List.of("r", "r", "r", "l", "l", "r", "l", "r", "l", "r"),
            Header.Y_HAT, List.of("", "", "", "", "", "", "", "", "", "")
    ));
    private final NavigableMap<Header, List<Double>> df = new TreeMap<>(Map.of(
            Header.X_1, List.of(3.0, 6.0, 0.0, -4.0, 1.0, 2.0, -4.0, 2.0, -9.0, 2.0),
            Header.X_2, List.of(2.0, -2.0, 1.0, -3.0, -9.0, 4.0, -2.0, 3.0, -3.0, 3.0),
            Header.X_3, List.of(5.0, -3.0, 0.0, -2.0, 2.0, -3.0, 5.0, 1.0, 2.0, -4.0),
            Header.Y, List.of(1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            Header.Y_HAT, List.of(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ));
    private final NavigableMap<Header, List<String>> regValidationDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("4", "2", "-2", "-9"),
            Header.X_2, List.of("1", "5", "-4", "2"),
            Header.X_3, List.of("3", "-4", "1", "4"),
            Header.Y, List.of("1.3", "1.4", "0.1", "0.2")
    ));
    private final NavigableMap<Header, List<String>> regTrainingDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4", "2", "-9", "2"),
            Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2", "3", "-3", "3"),
            Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5", "1", "2", "-4"),
            Header.Y, List.of("1.2", "1.3", "1.4", "0.8", "0.4", "1.5", "0.3", "1.8", "0.5", "1.3")
    ));
    private final NavigableMap<Header, List<Double>> regDf = new TreeMap<>(Map.of(
            Header.X_1, List.of(3.0, 6.0, 0.0, -4.0, 1.0, 2.0, -4.0, 2.0, -9.0, 2.0),
            Header.X_2, List.of(2.0, -2.0, 1.0, -3.0, -9.0, 4.0, -2.0, 3.0, -3.0, 3.0),
            Header.X_3, List.of(5.0, -3.0, 0.0, -2.0, 2.0, -3.0, 5.0, 1.0, 2.0, -4.0),
            Header.Y, List.of(1.2, 1.3, 1.4, 0.8, 0.4, 1.5, 0.3, 1.8, 0.5, 1.3),
            Header.Y_HAT, List.of(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ));
    private Network network;
    private Validator validator;
    private Trainer defaultTrainer;
    private Trainer trainer;
    private Network regNetwork;
    private Validator regValidator;
    private Trainer regTrainer;

    @BeforeEach
    void setUp() {
        network = new Network(3, 2, List.of(4, 3));
        validator = new Validator(network, validationDf, 1);
        defaultTrainer = new Trainer(network, trainingDf, 2, 5);
        trainer = new Trainer(network, trainingDf, 2, true,
                10,5, validator);
        regNetwork = new Network(3, 2, List.of(4, 3),
                2,0.01,0.01, true, false,
                Double.NaN,false);
        regValidator = new Validator(regNetwork, regValidationDf, 1);
        regTrainer = new Trainer(regNetwork, regTrainingDf, 2, 5);
    }

    @Override
    public Map<String, List<Object>> getLearnerWithExpectedAttributes() {
        Map<String, List<Object>> constructMap = new HashMap<>();
        constructMap.put("learners", List.of(defaultTrainer, trainer, regTrainer));
        constructMap.put("networks", List.of(network, network, regNetwork));
        constructMap.put("regressors", List.of(false, false, true));
        constructMap.put("numDatapoints", List.of(10, 10, 10));
        constructMap.put("batchSizes", List.of(2, 2, 2));
        constructMap.put("categoryNames", List.of(
                List.of("l", "r"), List.of("l", "r"), List.of()));
        constructMap.put("categoricalDfs", List.of(categoricalDf, categoricalDf, new TreeMap<>(Map.of())));
        constructMap.put("dfs", List.of(df, df, regDf));
        constructMap.put("weighted", List.of(false, true, false));
        return constructMap;
    }

    @Override
    public Map<String, List<Object>> getLearnerWithExpectedLoss() {
        List<List<Double>> outputNeuronVals = List.of(
                List.of(0.2, 0.8), List.of(0.7, 0.3));
        List<List<Double>> regOutputNeuronVals = List.of(
                List.of(1.9), List.of(0.2));
        Map<String, List<Object>> forwardPassMap = new HashMap<>();
        Network mockNetwork = mock(Network.class);
        doReturn(List.of(0.2, 0.8), List.of(0.7, 0.3)).when(mockNetwork)
                .forwardPassOneDatapoint(anyList());
        Network mockRegNetwork = mock(Network.class);
        doReturn(List.of(1.9), List.of(0.2)).when(mockRegNetwork)
                .forwardPassOneDatapoint(anyList());
        CrossEntropyLoss mockCrossEntropyLoss = mock(CrossEntropyLoss.class);
        doReturn(0.2, 0.3).when(mockCrossEntropyLoss)
                .call(anyList(), anyInt());
        MSELoss mockMSELoss = mock(MSELoss.class);
        doReturn(0.2, 0.3).when(mockMSELoss)
                .call(anyDouble(), anyDouble());
        Trainer spyTrainer = spy(
                new Trainer(network, trainingDf, 2, 5));
        spyTrainer.setNetwork(mockNetwork);
        spyTrainer.setCrossEntropyLoss(mockCrossEntropyLoss);
        Trainer spyRegTrainer = spy(
                new Trainer(regNetwork, regTrainingDf, 2, 5));
        spyRegTrainer.setNetwork(mockRegNetwork);
        spyRegTrainer.setMseLoss(mockMSELoss);
        forwardPassMap.put("learners", List.of(spyTrainer, spyRegTrainer));
        forwardPassMap.put("batchIds", List.of(List.of(7, 8), List.of(7, 8)));
        forwardPassMap.put("outputNeuronVals", List.of(outputNeuronVals, regOutputNeuronVals));
        forwardPassMap.put("totalLosses", List.of(0.5, 0.5));
        forwardPassMap.put("lossFunctions", List.of(mockCrossEntropyLoss, mockMSELoss));
        forwardPassMap.put("yHats", List.of(List.of(1, 0), List.of(1.8, 0.5)));
        forwardPassMap.put("predictedYHats", List.of(List.of(1.0, 0.0)));
        return forwardPassMap;
    }

    @Override
    public Map<String, List<Object>> getLearnerWithExpectedCategoricalDf() {
        Map<String, List<Object>> categoricalMap = new HashMap<>();
        categoricalMap.put("learners", List.of(trainer, regTrainer));
        NavigableMap<Header, List<String>> copyCategoricalDf = trainer.getCategoricalDf();
        copyCategoricalDf.put(Header.Y_HAT, Collections.nCopies(10, "l"));
        categoricalMap.put("dfs", List.of(copyCategoricalDf));
        return categoricalMap;
    }

    @Test
    void constructErroneous() {
        // 1. Not enough columns
        NavigableMap<Header, List<String>> badDf1 = new TreeMap<>(Map.of(
                Header.X_1, List.of("4", "2", "-2", "-9"),
                Header.X_2, List.of("1", "5", "-4", "2"),
                Header.Y, List.of("r", "r", "l", "l")
        ));
        Exception exception1 = assertThrows(IllegalArgumentException.class,
                () -> new Trainer(network, badDf1, 1, 5));
        assertEquals("Number of features must match number of initial " +
                "neurons (features = 2, initial neurons = 3)", exception1.getMessage());

        // 2. Too many output labels for the size of the network
        NavigableMap<Header, List<String>> badDf2 = new TreeMap<>(Map.of(
                Header.X_1, List.of("4", "2", "-2", "-9"),
                Header.X_2, List.of("1", "5", "-4", "2"),
                Header.X_3, List.of("3", "-4", "1", "4"),
                Header.Y, List.of("r", "u", "l", "l")
        ));
        Exception exception2 = assertThrows(IllegalArgumentException.class,
                () -> new Trainer(network, badDf2, 1, 5));
        assertEquals("The number of output neurons in the network " +
                "(2) is less than the number of " +
                "classes in the dataframe (3)", exception2.getMessage());

        // 3. Batch size too big
        Exception exception3 = assertThrows(IllegalArgumentException.class,
                () -> new Trainer(network, trainingDf, 11, 5));
        assertEquals("Batch size must be smaller than number of " +
                "datapoints", exception3.getMessage());
    }

    @Test
    void construct() {
        super.construct();
        Map<String, List<Double>> defaultLossDf = defaultTrainer.getLossDf();
        assertTrue(defaultLossDf.containsKey("Training"));
        assertFalse(defaultLossDf.containsKey("Validation"));
        assertIterableEquals(List.of(), defaultLossDf.get("Training"));
        Map<String, List<Double>> lossDf = trainer.getLossDf();
        for (String header : List.of("Training", "Validation")) {
            assertTrue(lossDf.containsKey(header));
            assertIterableEquals(List.of(), lossDf.get(header));
        }
    }

    @Test
    void storeGradients() {
        Network spyNetwork = spy(network);
        doNothing().when(spyNetwork).storeGradientOfLoss(
                any(Edge.class), anyInt(), anyBoolean());
        trainer.setNetwork(spyNetwork);
        int id = 4;
        trainer.storeGradients(id);
        List<List<List<Edge>>> edges = network.getEdges();
        for (List<List<Edge>> edgeLayer : edges) {
            for (List<Edge> rightNeuron : edgeLayer) {
                boolean first = true;
                for (Edge edge : rightNeuron) {
                    verify(spyNetwork, times(1))
                            .storeGradientOfLoss(edge, 0, first);
                    first = false;
                }
            }
        }
    }

    @Test
    void backPropagateOneBatch() {
        Network mockNetwork = mock(Network.class);
        trainer.setNetwork(mockNetwork);
        trainer.backPropagateOneBatch();
        verify(mockNetwork, times(1))
                .backPropagateWeights();
        verify(mockNetwork, times(1))
                .backPropagateBiases();
    }
}
