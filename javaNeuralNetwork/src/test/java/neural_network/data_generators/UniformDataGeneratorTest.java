package neural_network.data_generators;

import neural_network.util.Header;
import org.apache.commons.csv.CSVPrinter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.verify;

public class UniformDataGeneratorTest extends DataGeneratorTest {

    private UniformDataGenerator<Integer> oneCoordGen;
    private Method oneCoordMethod;
    private UniformDataGenerator<Double> twoCoordGen;
    private Method twoCoordMethod;
    private UniformDataGenerator<String> strGen;
    private Method strMethod;
    private final NavigableMap<Header, List<String>> oneCoordDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("0.8", "1.2", "1.1", "0.6", "-0.1"),
            Header.Y, List.of("4", "5", "5", "4", "3")));
    private final NavigableMap<Header, List<String>> twoCoordDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("-0.8", "-0.5", "1.4", "0.5", "0.2"),
            Header.X_2, List.of("0.5", "2.1", "-0.5", "4.2", "3.1"),
            Header.Y, List.of("-0.3", "1.6", "0.9", "4.7", "3.3")));
    private final NavigableMap<Header, List<String>> strDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("4.5", "6.2", "11.7", "7.6", "8.8"),
            Header.X_2, List.of("0.5", "-1.3", "2.4", "0.7", "-0.6"),
            Header.Y, List.of("Odd", "Even", "Even", "Even", "Even")));

    @BeforeEach
    void setUp() throws NoSuchMethodException {
        oneCoordMethod = DataGeneratorTest.class
                .getMethod("oneCoordClassifier", double.class);
        oneCoordGen = new UniformDataGenerator<>(
                oneCoordMethod, 5, List.of(-1.0), List.of(2.0));
        twoCoordMethod = DataGeneratorTest.class
                .getMethod("twoCoordRegressor", double.class, double.class);
        twoCoordGen = new UniformDataGenerator<>(
                twoCoordMethod,5, List.of(-2.0, -1.0), List.of(2.0, 5.0));
        strMethod = DataGeneratorTest.class
                .getMethod("stringClassifier", double.class, double.class);
        strGen = new UniformDataGenerator<>(
                strMethod, 5, List.of(4.0, -2.0), List.of(12.0, 3.0));
    }

    @Test
    void constructErroneous() {
        Exception exception1 = assertThrows(IllegalArgumentException.class,
                () -> new UniformDataGenerator<>(oneCoordMethod, 10,
                        List.of(1.0, 2.0), List.of(2.0)));
        assertEquals("The function method accepts 1 parameters but " +
                "we have 2 lower bounds.", exception1.getMessage());
        Exception exception2 = assertThrows(IllegalArgumentException.class,
                () -> new UniformDataGenerator<>(oneCoordMethod, 10,
                        List.of(0.0), List.of(1.0, 2.0)));
        assertEquals("The function method accepts 1 parameters but " +
                "we have 2 upper bounds.", exception2.getMessage());
        Exception exception3 = assertThrows(IllegalArgumentException.class,
                () -> new UniformDataGenerator<>(twoCoordMethod, 10,
                        List.of(1.0, -1.0), List.of(1.0, 0.0)));
        assertEquals("All lower bounds must be lower than their " +
                "related upper bounds (1.000000 >= 1.000000)", exception3.getMessage());
    }

    @Test
    void construct() {
        assertEquals(1, oneCoordGen.getDimensions());
        assertEquals(2, twoCoordGen.getDimensions());
        assertEquals(2, strGen.getDimensions());
        assertEquals(5, oneCoordGen.getNumDatapoints());
        assertEquals(5, twoCoordGen.getNumDatapoints());
        assertEquals(5, strGen.getNumDatapoints());
    }

    @Override
    public Map<String, List<Object>> getGeneratorWithExpectedDf() {
        Random oneCoordRandom = mock(Random.class);
        when(oneCoordRandom.nextDouble(-1.0, 2.0))
                .thenReturn(0.8, 1.2, 1.1, 0.6, -0.1);
        Random twoCoordRandom = mock(Random.class);
        when(twoCoordRandom.nextDouble(-2.0, 2.0))
                .thenReturn(-0.8, -0.5, 1.4, 0.5, 0.2);
        when(twoCoordRandom.nextDouble(-1.0, 5.0))
                .thenReturn(0.5, 2.1, -0.5, 4.2, 3.1);
        Random strRandom = mock(Random.class);
        when(strRandom.nextDouble(4.0, 12.0))
                .thenReturn(4.5, 6.2, 11.7, 7.6, 8.8);
        when(strRandom.nextDouble(-2.0, 3.0))
                .thenReturn(0.5, -1.3, 2.4, 0.7, -0.6);
        Map<String, List<Object>> callMap = new HashMap<>();
        callMap.put("generators", List.of(oneCoordGen, twoCoordGen, strGen));
        callMap.put("randoms", List.of(oneCoordRandom, twoCoordRandom, strRandom));
        callMap.put("dfs", List.of(oneCoordDf, twoCoordDf, strDf));
        return callMap;
    }

    @Test
    void writeToCsvErroneous() {
        Exception exception1 = assertThrows(RuntimeException.class,
                () -> oneCoordGen.writeToCsv("testing", "fake_dir"));
        assertEquals("Path fake_dir/testing.csv does not exist or is " +
                        "otherwise invalid.",
                exception1.getMessage());
        Exception exception2 = assertThrows(RuntimeException.class,
                () -> oneCoordGen.writeToCsv("/invalid_name"));
        assertEquals("Path /invalid_name.csv does not exist or is " +
                        "otherwise invalid.",
                exception2.getMessage());
    }

    @Test
    void writeToCsvOneCoord() throws InvocationTargetException, IllegalAccessException, IOException {
        oneCoordGen.call();
        if (Files.exists(Path.of("javaNeuralNetwork"))) {
            oneCoordGen.writeToCsv("testing",
                    "javaNeuralNetwork/src/test/resources/data_generators");
        } else {
            oneCoordGen.writeToCsv("testing",
                    "src/test/resources/data_generators");
        }
        CSVPrinter printer = oneCoordGen.getPrinter();
        assertNotNull(printer);
    }

    @Test
    void writeToCsvTwoCoord() throws InvocationTargetException, IllegalAccessException, IOException {
        Random twoCoordRandom = mock(Random.class);
        when(twoCoordRandom.nextDouble(-2.0, 2.0))
                .thenReturn(-0.8, -0.5, 1.4, 0.5, 0.2);
        when(twoCoordRandom.nextDouble(-1.0, 5.0))
                .thenReturn(0.5, 2.1, -0.5, 4.2, 3.1);
        DataGenerator.setRandom(twoCoordRandom);
        twoCoordGen.call();
        CSVPrinter mockPrinter = mock(CSVPrinter.class);
        twoCoordGen.setPrinter(mockPrinter);
        twoCoordGen.writeToCsv("testing2", "javaNeuralNetwork/src/test/resources/data_generators");
        List<Header> headers = new ArrayList<>(List.of(Header.X_1, Header.X_2, Header.Y));
        verify(mockPrinter).printRecord(headers);
        verify(mockPrinter).printRecord(List.of("-0.8", "0.5", "-0.3"));
        verify(mockPrinter).printRecord(List.of("-0.5", "2.1", "1.6"));
        verify(mockPrinter).printRecord(List.of("1.4", "-0.5", "0.9"));
        verify(mockPrinter).printRecord(List.of("0.5", "4.2", "4.7"));
        verify(mockPrinter).printRecord(List.of("0.2", "3.1", "3.3"));
        verify(mockPrinter).flush();
    }
}
