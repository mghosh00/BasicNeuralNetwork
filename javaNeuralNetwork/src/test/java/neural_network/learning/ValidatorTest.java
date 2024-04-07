package neural_network.learning;

import neural_network.components.Network;
import neural_network.functions.CrossEntropyLoss;
import neural_network.functions.MSELoss;
import neural_network.util.Header;
import neural_network.util.Partitioner;
import neural_network.util.Plotter;
import neural_network.util.WeightedPartitioner;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.doNothing;

public class ValidatorTest extends LearnerTest {
    private final List<List<Integer>> partitions = List.of(
            List.of(2), List.of(0), List.of(1), List.of(3));
    private final List<List<Integer>> weightedPartitions = List.of(
            List.of(2), List.of(2), List.of(1), List.of(0));
    private final List<Double> batchLosses = List.of(
            0.4, 0.2, 0.5, 0.1);
    private final NavigableMap<Header, List<String>> validationDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("4", "2", "-2", "-9"),
            Header.X_2, List.of("1", "5", "-4", "2"),
            Header.X_3, List.of("3", "-4", "1", "4"),
            Header.Y, List.of("r", "r", "l", "l")
    ));
    private final NavigableMap<Header, List<String>> categoricalDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("4", "2", "-2", "-9"),
            Header.X_2, List.of("1", "5", "-4", "2"),
            Header.X_3, List.of("3", "-4", "1", "4"),
            Header.Y, List.of("r", "r", "l", "l"),
            Header.Y_HAT, List.of("", "", "", "")
    ));
    private final NavigableMap<Header, List<Double>> df = new TreeMap<>(Map.of(
            Header.X_1, List.of(4.0, 2.0, -2.0, -9.0),
            Header.X_2, List.of(1.0, 5.0, -4.0, 2.0),
            Header.X_3, List.of(3.0, -4.0, 1.0, 4.0),
            Header.Y, List.of(1.0, 1.0, 0.0, 0.0),
            Header.Y_HAT, List.of(0.0, 0.0, 0.0, 0.0)
    ));
    private final NavigableMap<Header, List<String>> regValidationDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("4", "2", "-2", "-9"),
            Header.X_2, List.of("1", "5", "-4", "2"),
            Header.X_3, List.of("3", "-4", "1", "4"),
            Header.Y, List.of("1.3", "1.4", "0.1", "0.2")
    ));
    private final NavigableMap<Header, List<Double>> regDf = new TreeMap<>(Map.of(
            Header.X_1, List.of(4.0, 2.0, -2.0, -9.0),
            Header.X_2, List.of(1.0, 5.0, -4.0, 2.0),
            Header.X_3, List.of(3.0, -4.0, 1.0, 4.0),
            Header.Y, List.of(1.3, 1.4, 0.1, 0.2),
            Header.Y_HAT, List.of(0.0, 0.0, 0.0, 0.0)
    ));
    private Network network;
    private Validator defaultValidator;
    private Validator validator;
    private Network regNetwork;
    private Validator regValidator;
    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;

    @BeforeEach
    public void setUpStreams() {
        System.setOut(new PrintStream(outContent));
    }

    @AfterEach
    public void restoreStreams() {
        System.setOut(originalOut);
    }

    @BeforeEach
    void setUp() {
        network = new Network(3, 2, List.of(4, 3));
        defaultValidator = new Validator(network, validationDf, 1);
        validator = new Validator(network, validationDf, 1,
                true, 10);
        regNetwork = new Network(3, 2, List.of(4, 3),
                2,0.01,0.01, true, false,
                Double.NaN,false);
        regValidator = new Validator(regNetwork, regValidationDf, 1);
    }

    @Override
    public Map<String, List<Object>> getLearnerWithExpectedAttributes() {
        Map<String, List<Object>> constructMap = new HashMap<>();
        constructMap.put("learners", List.of(defaultValidator, validator, regValidator));
        constructMap.put("networks", List.of(network, network, regNetwork));
        constructMap.put("regressors", List.of(false, false, true));
        constructMap.put("numDatapoints", List.of(4, 4, 4));
        constructMap.put("batchSizes", List.of(1, 1, 1));
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
                List.of(0.2, 0.8));
        List<List<Double>> regOutputNeuronVals = List.of(
                List.of(1.9));
        Map<String, List<Object>> forwardPassMap = new HashMap<>();
        Network mockNetwork = mock(Network.class);
        doReturn(List.of(0.2, 0.8)).when(mockNetwork)
                .forwardPassOneDatapoint(anyList());
        Network mockRegNetwork = mock(Network.class);
        doReturn(List.of(1.9)).when(mockRegNetwork)
                .forwardPassOneDatapoint(anyList());
        CrossEntropyLoss mockCrossEntropyLoss = mock(CrossEntropyLoss.class);
        doReturn(0.2).when(mockCrossEntropyLoss)
                .call(anyList(), anyInt());
        MSELoss mockMSELoss = mock(MSELoss.class);
        doReturn(0.2).when(mockMSELoss)
                .call(anyDouble(), anyDouble());
        Validator spyValidator = spy(
                new Validator(network, validationDf, 1));
        spyValidator.setNetwork(mockNetwork);
        spyValidator.setCrossEntropyLoss(mockCrossEntropyLoss);
        Validator spyRegValidator = spy(
                new Validator(regNetwork, regValidationDf, 1));
        spyRegValidator.setNetwork(mockRegNetwork);
        spyRegValidator.setMseLoss(mockMSELoss);
        forwardPassMap.put("learners", List.of(spyValidator, spyRegValidator));
        forwardPassMap.put("batchIds", List.of(List.of(2), List.of(2)));
        forwardPassMap.put("outputNeuronVals", List.of(outputNeuronVals, regOutputNeuronVals));
        forwardPassMap.put("totalLosses", List.of(0.2, 0.2));
        forwardPassMap.put("lossFunctions", List.of(mockCrossEntropyLoss, mockMSELoss));
        forwardPassMap.put("yHats", List.of(List.of(0), List.of(0.1)));
        forwardPassMap.put("predictedYHats", List.of(List.of(1.0)));
        return forwardPassMap;
    }

    @Override
    public Map<String, List<Object>> getLearnerWithExpectedCategoricalDf() {
        Map<String, List<Object>> categoricalMap = new HashMap<>();
        categoricalMap.put("learners", List.of(validator, regValidator));
        NavigableMap<Header, List<String>> copyCategoricalDf = validator.getCategoricalDf();
        copyCategoricalDf.put(Header.Y_HAT, Collections.nCopies(4, "l"));
        categoricalMap.put("dfs", List.of(copyCategoricalDf));
        return categoricalMap;
    }

    @Test
    void storeGradients() {
        Validator spyValidator = spy(validator);
        spyValidator.storeGradients(5);
        verify(spyValidator, times(1))
                .storeGradients(5);
    }

    @Test
    void validateDefault() {
        // Mock most of the validator methods and the partitioner call
        Validator spyValidator = spy(defaultValidator);
        Partitioner mockPartitioner = mock(Partitioner.class);
        when(mockPartitioner.call()).thenReturn(partitions);
        spyValidator.setPartitioner(mockPartitioner);
        doReturn(0.4, 0.2, 0.5, 0.1)
                .when(spyValidator).forwardPassOneBatch(anyList());
        doNothing().when(spyValidator).updateCategoricalDataframe();
        // Validate
        spyValidator.validate(1);
        // Ensure the partitioner and forward propagate are called the
        // correct number of times
        verify(mockPartitioner, times(1)).call();
        for (int i = 0; i < 4; i ++) {
            verify(spyValidator, times(1))
                    .forwardPassOneBatch(partitions.get(i));
        }
        // Now check the print calls
        assertTrue(outContent.toString().contains("Validation loss: 0.3000"));
        // Now validate again and check that this time, the epoch has increased,
        // and we get no print for epoch 1
        doReturn(0.1, 0.1, 0.1, 0.1)
                .when(spyValidator).forwardPassOneBatch(anyList());
        spyValidator.validate(2);
        assertTrue(outContent.toString().contains("Validation loss: 0.3000"));
        assertFalse(outContent.toString().contains("Validation loss: 0.1000"));
        verify(spyValidator, times(2))
                .updateCategoricalDataframe();
    }

    @Test
    void validate() {
        // Mock most of the validator methods and the partitioner call
        Validator spyValidator = spy(validator);
        WeightedPartitioner mockPartitioner = mock(WeightedPartitioner.class);
        when(mockPartitioner.call()).thenReturn(weightedPartitions);
        spyValidator.setPartitioner(mockPartitioner);
        doReturn(0.4, 0.2, 0.5, 0.1)
                .when(spyValidator).forwardPassOneBatch(anyList());
        doNothing().when(spyValidator).updateCategoricalDataframe();
        // Validate
        spyValidator.validate(1);
        // Ensure the partitioner and forward propagate are called the
        // correct number of times
        verify(mockPartitioner, times(1)).call();
        verify(spyValidator, times(1))
                .forwardPassOneBatch(List.of(0));
        verify(spyValidator, times(1))
                .forwardPassOneBatch(List.of(1));
        verify(spyValidator, times(2))
                .forwardPassOneBatch(List.of(2));
        verify(spyValidator, times(1))
                .updateCategoricalDataframe();
    }

    @Test
    void validateRegressor() {
        // Just check that the update categorical dataframe method
        // is not called
        Validator spyValidator = spy(regValidator);
        spyValidator.validate(1);
        verify(spyValidator, times(0))
                .updateCategoricalDataframe();
    }

    @Test
    void run() {
        Validator spyValidator = spy(validator);
        spyValidator.run();
        verify(spyValidator, times(1))
                .run();
    }

    @Test
    void generateScatter() throws IOException {
        try (MockedStatic<Plotter> mockPlotter = mockStatic(Plotter.class)) {
            validator.generateScatter("test_title");
            mockPlotter.verify(
                    () -> Plotter.datapointScatter(categoricalDf, "validation",
                            "test_title", false), times(1));
        }
    }

    @Test
    void comparisonScatter() throws IOException {
        Exception exception = assertThrows(RuntimeException.class,
                () -> validator.comparisonScatter("test_title"));
        assertEquals("Cannot call this method with categorical data.",
                exception.getMessage());
        try (MockedStatic<Plotter> mockPlotter = mockStatic(Plotter.class)) {
            regValidator.comparisonScatter("test_title");
            mockPlotter.verify(
                    () -> Plotter.comparisonScatter(regDf, "validation",
                            "test_title"), times(1));
        }
    }
}
