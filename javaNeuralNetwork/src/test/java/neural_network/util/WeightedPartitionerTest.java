package neural_network.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.*;

public class WeightedPartitionerTest {

    // mockRandomChoice is for reference only
    private final List<Integer> mockRandomChoice = List.of(3, 4, 1, 8, 3, 6, 6, 9, 2, 5);
    private final IntStream mockRandomStream = IntStream.of(2, 0, 1, 0, 2, 0, 0, 1, 2, 1);
    private final List<Integer> ints = List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    private final List<Double> yVals = List.of(1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    private final List<Double> regYVals = List.of(1.4, 6.2, 5.3, 8.8, 3.4, 3.1, 2.6, 1.0, 9.0, 8.2);
    private WeightedPartitioner evenPartitioner;
    private WeightedPartitioner unevenPartitioner;
    private WeightedPartitioner bigPartitioner;
    private WeightedPartitioner smallPartitioner;
    private WeightedPartitioner regPartitioner;
    private Random mockRandom;

    @BeforeEach
    void setUp() {
        mockRandom = mock(Random.class);
        when(mockRandom.ints(10, 0, 3))
                .thenReturn(mockRandomStream);
        when(mockRandom.nextInt(anyInt()))
                .thenReturn(1, 0, 1, 2, 1, 1, 1, 4, 0, 2);
        evenPartitioner = new WeightedPartitioner(10, 5, yVals);
        evenPartitioner.setRandom(mockRandom);
        unevenPartitioner = new WeightedPartitioner(10, 3, yVals);
        unevenPartitioner.setRandom(mockRandom);
        bigPartitioner = new WeightedPartitioner(10, 10, yVals);
        bigPartitioner.setRandom(mockRandom);
        smallPartitioner = new WeightedPartitioner(10, 1, yVals);
        smallPartitioner.setRandom(mockRandom);
        regPartitioner = new WeightedPartitioner(10, 5, regYVals,
                true, 8);
    }

    @Test
    void constructErroneous() {
        Exception exception = assertThrows(IllegalArgumentException.class,
                () -> new WeightedPartitioner(8, 5, yVals));
        assertEquals("numInts must equal the length of the yVals " +
                "(numInts = 8, yVals.size() = 10)", exception.getMessage());
    }

    @Test
    void construct() {
        assertEquals(10, evenPartitioner.getNumInts());
        assertEquals(5, evenPartitioner.getSetSize());
        Map<Integer, List<Integer>> expectedMap = Map.of(0, List.of(4, 6, 8),
                1, List.of(0, 1, 5, 7, 9), 2, List.of(2, 3));
        for (int k = 0; k <= 2; k ++) {
            Map<Integer, List<Integer>> classMap = evenPartitioner.getClassMap();
            assertTrue(classMap.containsKey(k));
            assertIterableEquals(expectedMap.get(k), classMap.get(k));
        }
    }

    @Test
    void constructRegression() {
        assertEquals(10, regPartitioner.getNumInts());
        assertEquals(5, regPartitioner.getSetSize());
        Map<Integer, List<Integer>> expectedMap = Map.of(0, List.of(0, 7),
                1, List.of(6), 2, List.of(4, 5), 3, List.of(2),
                4, List.of(1), 5, List.of(3, 8, 9));
        for (int k = 0; k <= 5; k ++) {
            Map<Integer, List<Integer>> classMap = regPartitioner.getClassMap();
            assertTrue(classMap.containsKey(k));
            assertIterableEquals(expectedMap.get(k), classMap.get(k));
        }
    }

    @Test
    void callEvenPartitioner() {
        assertIterableEquals(List.of(List.of(3, 4, 1, 8, 3), List.of(6, 6, 9, 2, 5)),
                evenPartitioner.call());
        verify(mockRandom, times(1))
                .ints(10, 0, 3);
        verify(mockRandom, times(3))
                .nextInt(2);
        verify(mockRandom, times(4))
                .nextInt(3);
        verify(mockRandom, times(3))
                .nextInt(5);
    }

    @Test
    void callUnevenPartitioner() {
        assertIterableEquals(List.of(List.of(3, 4, 1), List.of(8, 3, 6), List.of(6, 9, 2),
                List.of(5)),
                unevenPartitioner.call());
        verify(mockRandom, times(1))
                .ints(10, 0, 3);
        verify(mockRandom, times(3))
                .nextInt(2);
        verify(mockRandom, times(4))
                .nextInt(3);
        verify(mockRandom, times(3))
                .nextInt(5);
    }

    @Test
    void callBigPartitioner() {
        assertIterableEquals(List.of(mockRandomChoice),
                bigPartitioner.call());
        verify(mockRandom, times(1))
                .ints(10, 0, 3);
        verify(mockRandom, times(3))
                .nextInt(2);
        verify(mockRandom, times(4))
                .nextInt(3);
        verify(mockRandom, times(3))
                .nextInt(5);
    }

    @Test
    void callSmallPartitioner() {
        assertIterableEquals(List.of(List.of(3), List.of(4), List.of(1),
                        List.of(8), List.of(3), List.of(6),
                        List.of(6), List.of(9), List.of(2), List.of(5)),
                smallPartitioner.call());
        verify(mockRandom, times(1))
                .ints(10, 0, 3);
        verify(mockRandom, times(3))
                .nextInt(2);
        verify(mockRandom, times(4))
                .nextInt(3);
        verify(mockRandom, times(3))
                .nextInt(5);
    }
}
