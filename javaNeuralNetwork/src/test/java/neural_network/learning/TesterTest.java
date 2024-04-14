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
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

public class TesterTest extends LearnerTest {
    private final List<List<Integer>> partitions = List.of(
            List.of(2, 4, 0), List.of(3, 1));
    private final List<List<Integer>> weightedPartitions = List.of(
            List.of(1, 3, 2), List.of(0, 4, 3));
    private final List<Double> batchLosses = List.of(
            0.5, 0.2);
    private final NavigableMap<Header, List<String>> testingDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("-2", "2", "-8", "-8", "2"),
            Header.X_2, List.of("0", "6", "-2", "4", "6"),
            Header.X_3, List.of("3", "-9", "9", "1", "-8"),
            Header.Y, List.of("r", "r", "l", "l", "l")
    ));
    private final NavigableMap<Header, List<String>> categoricalDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("-2", "2", "-8", "-8", "2"),
            Header.X_2, List.of("0", "6", "-2", "4", "6"),
            Header.X_3, List.of("3", "-9", "9", "1", "-8"),
            Header.Y, List.of("r", "r", "l", "l", "l"),
            Header.Y_HAT, List.of("", "", "", "", "")
    ));
    private final NavigableMap<Header, List<Double>> df = new TreeMap<>(Map.of(
            Header.X_1, List.of(-2.0, 2.0, -8.0, -8.0, 2.0),
            Header.X_2, List.of(0.0, 6.0, -2.0, 4.0, 6.0),
            Header.X_3, List.of(3.0, -9.0, 9.0, 1.0, -8.0),
            Header.Y, List.of(1.0, 1.0, 0.0, 0.0, 0.0),
            Header.Y_HAT, List.of(0.0, 0.0, 0.0, 0.0, 0.0)
    ));
    private final NavigableMap<Header, List<String>> regTestingDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("-2", "2", "-8", "-8", "2"),
            Header.X_2, List.of("0", "6", "-2", "4", "6"),
            Header.X_3, List.of("3", "-9", "9", "1", "-8"),
            Header.Y, List.of("1.0", "1.3", "0.4", "0.2", "0.3")
    ));
    private final NavigableMap<Header, List<Double>> regDf = new TreeMap<>(Map.of(
            Header.X_1, List.of(-2.0, 2.0, -8.0, -8.0, 2.0),
            Header.X_2, List.of(0.0, 6.0, -2.0, 4.0, 6.0),
            Header.X_3, List.of(3.0, -9.0, 9.0, 1.0, -8.0),
            Header.Y, List.of(1.0, 1.3, 0.4, 0.2, 0.3),
            Header.Y_HAT, List.of(0.0, 0.0, 0.0, 0.0, 0.0)
    ));
    private Network network;
    private Tester defaultTester;
    private Tester tester;
    private Network regNetwork;
    private Tester regTester;
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
        defaultTester = new Tester(network, testingDf, 3);
        tester = new Tester(network, testingDf, 3,
                true, 10);
        regNetwork = new Network(3, 2, List.of(4, 3),
                2,0.01,0.01, true, false,
                Double.NaN,false);
        regTester = new Tester(regNetwork, regTestingDf, 3);
    }

    @Override
    public Map<String, List<Object>> getLearnerWithExpectedAttributes() {
        Map<String, List<Object>> constructMap = new HashMap<>();
        constructMap.put("learners", List.of(defaultTester, tester, regTester));
        constructMap.put("networks", List.of(network, network, regNetwork));
        constructMap.put("regressors", List.of(false, false, true));
        constructMap.put("numDatapoints", List.of(5, 5, 5));
        constructMap.put("batchSizes", List.of(3, 3, 3));
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
        Tester spyTester = spy(
                new Tester(network, testingDf, 1));
        spyTester.setNetwork(mockNetwork);
        spyTester.setCrossEntropyLoss(mockCrossEntropyLoss);
        Tester spyRegTester = spy(
                new Tester(regNetwork, regTestingDf, 1));
        spyRegTester.setNetwork(mockRegNetwork);
        spyRegTester.setMseLoss(mockMSELoss);
        forwardPassMap.put("learners", List.of(spyTester, spyRegTester));
        forwardPassMap.put("batchIds", List.of(List.of(2), List.of(2)));
        forwardPassMap.put("outputNeuronVals", List.of(outputNeuronVals, regOutputNeuronVals));
        forwardPassMap.put("totalLosses", List.of(0.2, 0.2));
        forwardPassMap.put("lossFunctions", List.of(mockCrossEntropyLoss, mockMSELoss));
        forwardPassMap.put("yHats", List.of(List.of(0), List.of(0.4)));
        forwardPassMap.put("predictedYHats", List.of(List.of(1.0)));
        return forwardPassMap;
    }

    @Override
    public Map<String, List<Object>> getLearnerWithExpectedCategoricalDf() {
        Map<String, List<Object>> categoricalMap = new HashMap<>();
        categoricalMap.put("learners", List.of(tester, regTester));
        NavigableMap<Header, List<String>> copyCategoricalDf = tester.getCategoricalDf();
        copyCategoricalDf.put(Header.Y_HAT, Collections.nCopies(5, "l"));
        categoricalMap.put("dfs", List.of(copyCategoricalDf));
        return categoricalMap;
    }

    @Test
    void storeGradients() {
        Tester spyTester = spy(tester);
        spyTester.storeGradients(5);
        verify(spyTester, times(1))
                .storeGradients(5);
    }

    @Test
    void runDefault() {
        // Mock most of the tester methods and the partitioner call
        Tester spyTester = spy(defaultTester);
        Partitioner mockPartitioner = mock(Partitioner.class);
        when(mockPartitioner.call()).thenReturn(partitions);
        spyTester.setPartitioner(mockPartitioner);
        doReturn(0.5, 0.2)
                .when(spyTester).forwardPassOneBatch(anyList());
        doNothing().when(spyTester).updateCategoricalDataframe();
        // Run
        spyTester.run();
        // Ensure the partitioner and forward propagate are called the
        // correct number of times
        verify(mockPartitioner, times(1)).call();
        for (int i = 0; i < 2; i ++) {
            verify(spyTester, times(1))
                    .forwardPassOneBatch(partitions.get(i));
        }
        // Now check the print calls
        assertTrue(outContent.toString().contains("Testing loss: 0.1400"));
        // Now validate again and check that this time, the epoch has increased,
        // and we get no print for epoch 1
        verify(spyTester, times(1))
                .updateCategoricalDataframe();
    }

    @Test
    void run() {
        // Mock most of the tester methods and the partitioner call
        Tester spyTester = spy(tester);
        WeightedPartitioner mockPartitioner = mock(WeightedPartitioner.class);
        when(mockPartitioner.call()).thenReturn(weightedPartitions);
        spyTester.setPartitioner(mockPartitioner);
        doReturn(0.5, 0.2)
                .when(spyTester).forwardPassOneBatch(anyList());
        doNothing().when(spyTester).updateCategoricalDataframe();
        // Run
        spyTester.run();
        // Ensure the partitioner and forward propagate are called the
        // correct number of times
        verify(mockPartitioner, times(1)).call();
        for (int i = 0; i < 2; i ++) {
            verify(spyTester, times(1))
                    .forwardPassOneBatch(weightedPartitions.get(i));
        }
        verify(spyTester, times(1))
                .updateCategoricalDataframe();
    }

    @Test
    void runRegressor() {
        // Just check that the update categorical dataframe method
        // is not called
        Tester spyTester = spy(regTester);
        spyTester.run();
        verify(spyTester, times(0))
                .updateCategoricalDataframe();
    }

    @Test
    void generateScatter() throws IOException {
        try (MockedStatic<Plotter> mockPlotter = mockStatic(Plotter.class)) {
            tester.generateScatter("test_title");
            mockPlotter.verify(
                    () -> Plotter.datapointScatter(categoricalDf, "testing",
                            "test_title", false), times(1));
        }
    }

    @Test
    void comparisonScatter() throws IOException {
        Exception exception = assertThrows(RuntimeException.class,
                () -> tester.comparisonScatter("test_title"));
        assertEquals("Cannot call this method with categorical data.",
                exception.getMessage());
        try (MockedStatic<Plotter> mockPlotter = mockStatic(Plotter.class)) {
            regTester.comparisonScatter("test_title");
            mockPlotter.verify(
                    () -> Plotter.comparisonScatter(regDf, "testing",
                            "test_title"), times(1));
        }
    }

    @Test
    void generateConfusion() {
        Tester spyTester = spy(tester);
        doNothing().when(spyTester).printConfusion(anyList());
        doNothing().when(spyTester).printDiceScores(anyList());
        spyTester.setYHat(List.of("r", "l", "l", "l", "l"));
        spyTester.generateConfusion();
        List<List<Integer>> expectedContingencyTable = List.of(
                List.of(3, 0), List.of(1, 1));
        verify(spyTester, times(1))
                .printConfusion(expectedContingencyTable);
        verify(spyTester, times(1))
                .printDiceScores(expectedContingencyTable);
    }

    @Test
    void printConfusion() {
        List<List<Integer>> expectedContingencyTable = List.of(
                List.of(3, 0), List.of(1, 1));
        tester.printConfusion(expectedContingencyTable);
        String outputString = outContent.toString();
        assertTrue(outputString.contains("Confusion matrix"));
        assertTrue(outputString.contains("-----------------------------------------------"));
        assertTrue(outputString.contains("             yHat"));
        assertTrue(outputString.contains("             l  r"));
        assertTrue(outputString.contains("     l       3 0"));
        assertTrue(outputString.contains("y    r       1 1"));
    }

    @Test
    void printDiceScores() {
        List<List<Integer>> expectedContingencyTable = List.of(
                List.of(3, 0), List.of(1, 1));
        tester.printDiceScores(expectedContingencyTable);
        String outputString = outContent.toString();
        assertTrue(outputString.contains("Dice scores: {l=%.4f, r=%.4f}"
                .formatted(6.0 / 7, 2.0 / 3)));
        assertTrue(outputString.contains("Mean dice score: %.4f"
                .formatted(16.0 / 21)));
    }
}
