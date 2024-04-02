package neural_network.util;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class DataSplitterTest {

    private final NavigableMap<Header, List<String>> df = new TreeMap<>(Map.of(
            Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4", "2", "-9", "2"),
            Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2", "3", "-3", "3"),
            Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5", "1", "2", "-4"),
            Header.Y, List.of("1", "1", "1", "0", "0", "1", "0", "1", "0", "1")
    ));

    private DataSplitter splitter1;
    private DataSplitter splitter2;
    private DataSplitter splitter3;
    private DataSplitter splitter4;
    private DataSplitter splitter5;

    @BeforeEach
    void setUp() {
        splitter1 = spy(new DataSplitter("mock_path", List.of(8, 1, 1)));
        doReturn(df).when(splitter1).csvToMap();
        splitter2 = spy(new DataSplitter("mock_path", List.of(80, 20)));
        doReturn(df).when(splitter2).csvToMap();
        splitter3 = spy(new DataSplitter("mock_path", List.of(12, 2, 2)));
        doReturn(df).when(splitter3).csvToMap();
        splitter4 = spy(new DataSplitter("mock_path", List.of(7)));
        doReturn(df).when(splitter4).csvToMap();
        splitter5 = spy(new DataSplitter("mock_path", List.of(14, 1, 1)));
        doReturn(df).when(splitter5).csvToMap();
    }

    @Test
    void constructErroneous() {
        Exception exception1 = assertThrows(IllegalArgumentException.class,
                () -> new DataSplitter("mock_path", List.of()));
        assertEquals("proportions must have 1-3 elements denoting the " +
                "train:validation:test ratio", exception1.getMessage());
        Exception exception2 = assertThrows(IllegalArgumentException.class,
                () -> new DataSplitter("mock_path", List.of(3, 4, 2, 5)));
        assertEquals("proportions must have 1-3 elements denoting the " +
                "train:validation:test ratio", exception2.getMessage());
    }

    void writeTestCsv() throws IOException {
        // First construct a printer
        CSVPrinter printer = new CSVPrinter(new FileWriter("testing.csv"), CSVFormat.DEFAULT);
        // Print headers
        Header.setDimensions(3);
        List<Header> headers = Header.getInitialHeaders();
        printer.printRecord(headers);
        // Print datapoints
        for (int j = 0; j < 10; j ++) {
            int finalJ = j;
            List<String> record = Stream
                    .iterate(0, i -> i < 4, i -> i + 1)
                    .map(i -> df.get(headers.get(i)).get(finalJ))
                    .toList();
            printer.printRecord(record);
        }
        // Flush and close
        printer.flush();
        printer.close();
    }

    @Test
    void csvToMapErroneous() throws IOException {
        DataSplitter badSplitter1 = new DataSplitter("invalid_file.csv", List.of(4));
        Exception exception1 = assertThrows(RuntimeException.class,
                badSplitter1::csvToMap);
        assertEquals("Path invalid_file.csv is invalid.",
                exception1.getMessage());
    }

    @Test
    void csvToMap() throws IOException {
        writeTestCsv();
        DataSplitter goodSplitter = new DataSplitter("testing.csv", List.of(8, 1, 1));

        // Spy on the CSVFormat object to verify correct behaviour
        CSVFormat csvFormat = spy(CSVFormat.DEFAULT.builder()
                .build());
        goodSplitter.setCsvFormat(csvFormat);
        NavigableMap<Header, List<String>> actualMap = goodSplitter.csvToMap();
        verify(csvFormat).parse(any(Reader.class));

        for (Header header : df.keySet()) {
            assertTrue(actualMap.containsKey(header));
            assertIterableEquals(df.get(header), actualMap.get(header));
        }

        assertEquals(csvFormat, goodSplitter.getCsvFormat());

        // Do not spy on CSVFormat to check that the correct one is produced
        goodSplitter.setCsvFormat(null);
        NavigableMap<Header, List<String>> actualMap2 = goodSplitter.csvToMap();

        for (Header header : df.keySet()) {
            assertTrue(actualMap2.containsKey(header));
            assertIterableEquals(df.get(header), actualMap2.get(header));
        }

        assertNotEquals(csvFormat, goodSplitter.getCsvFormat());

        Files.delete(Path.of("testing.csv"));
    }

    @Test
    void split1() {
        // Tests an even split with training, validation and testing
        NavigableMap<Header, List<String>> trainingDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4", "2"),
                Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2", "3"),
                Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5", "1"),
                Header.Y, List.of("1", "1", "1", "0", "0", "1", "0", "1")
        ));
        NavigableMap<Header, List<String>> validationDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("-9"),
                Header.X_2, List.of("-3"),
                Header.X_3, List.of("2"),
                Header.Y, List.of("0")
        ));
        NavigableMap<Header, List<String>> testingDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("2"),
                Header.X_2, List.of("3"),
                Header.X_3, List.of("-4"),
                Header.Y, List.of("1")
        ));
        List<NavigableMap<Header, List<String>>> actualDfs = splitter1.split();
        Header.setDimensions(3);
        for (Header header : Header.getInitialHeaders()) {
            assertIterableEquals(trainingDf.get(header), actualDfs.get(0).get(header));
            assertIterableEquals(validationDf.get(header), actualDfs.get(1).get(header));
            assertIterableEquals(testingDf.get(header), actualDfs.get(2).get(header));
        }
    }

    @Test
    void split2() {
        // Tests an even split with training and validation
        NavigableMap<Header, List<String>> trainingDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4", "2"),
                Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2", "3"),
                Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5", "1"),
                Header.Y, List.of("1", "1", "1", "0", "0", "1", "0", "1")
        ));
        NavigableMap<Header, List<String>> validationDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("-9", "2"),
                Header.X_2, List.of("-3", "3"),
                Header.X_3, List.of("2", "-4"),
                Header.Y, List.of("0", "1")
        ));
        List<NavigableMap<Header, List<String>>> actualDfs = splitter2.split();
        Header.setDimensions(3);
        for (Header header : Header.getInitialHeaders()) {
            assertIterableEquals(trainingDf.get(header), actualDfs.get(0).get(header));
            assertIterableEquals(validationDf.get(header), actualDfs.get(1).get(header));
        }
    }

    @Test
    void split3() {
        // Tests an uneven split with training, validation and testing
        NavigableMap<Header, List<String>> trainingDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4"),
                Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2"),
                Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5"),
                Header.Y, List.of("1", "1", "1", "0", "0", "1", "0")
        ));
        NavigableMap<Header, List<String>> validationDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("2"),
                Header.X_2, List.of("3"),
                Header.X_3, List.of("1"),
                Header.Y, List.of("1")
        ));
        NavigableMap<Header, List<String>> testingDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("-9", "2"),
                Header.X_2, List.of("-3", "3"),
                Header.X_3, List.of("2", "-4"),
                Header.Y, List.of("0", "1")
        ));
        List<NavigableMap<Header, List<String>>> actualDfs = splitter3.split();
        Header.setDimensions(3);
        for (Header header : Header.getInitialHeaders()) {
            assertIterableEquals(trainingDf.get(header), actualDfs.get(0).get(header));
            assertIterableEquals(validationDf.get(header), actualDfs.get(1).get(header));
            assertIterableEquals(testingDf.get(header), actualDfs.get(2).get(header));
        }
    }

    @Test
    void split4() {
        // Tests with just a training set
        List<NavigableMap<Header, List<String>>> actualDfs = splitter4.split();
        Header.setDimensions(3);
        for (Header header : Header.getInitialHeaders()) {
            assertIterableEquals(df.get(header), actualDfs.get(0).get(header));
        }
    }

    @Test
    void split5() {
        // Tests that we cannot get dfs of size 0
        NavigableMap<Header, List<String>> trainingDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("3", "6", "0", "-4", "1", "2", "-4", "2"),
                Header.X_2, List.of("2", "-2", "1", "-3", "-9", "4", "-2", "3"),
                Header.X_3, List.of("5", "-3", "0", "-2", "2", "-3", "5", "1"),
                Header.Y, List.of("1", "1", "1", "0", "0", "1", "0", "1")
        ));
        NavigableMap<Header, List<String>> validationDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("-9"),
                Header.X_2, List.of("-3"),
                Header.X_3, List.of("2"),
                Header.Y, List.of("0")
        ));
        NavigableMap<Header, List<String>> testingDf = new TreeMap<>(Map.of(
                Header.X_1, List.of("2"),
                Header.X_2, List.of("3"),
                Header.X_3, List.of("-4"),
                Header.Y, List.of("1")
        ));
        List<NavigableMap<Header, List<String>>> actualDfs = splitter5.split();
        Header.setDimensions(3);
        for (Header header : Header.getInitialHeaders()) {
            assertIterableEquals(trainingDf.get(header), actualDfs.get(0).get(header));
            assertIterableEquals(validationDf.get(header), actualDfs.get(1).get(header));
            assertIterableEquals(testingDf.get(header), actualDfs.get(2).get(header));
        }
    }
}
