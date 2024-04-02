package neural_network.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class PartitionerTest {

    private final List<Integer> ints = new ArrayList<>(
            List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    // The below is from seed 42 of the Random class
    private final List<Integer> shuffledInts = new ArrayList<>(
            List.of(4, 6, 2, 1, 7, 9, 8, 5, 3, 0));
    private Partitioner evenPartitioner;
    private Partitioner unevenPartitioner;
    private Partitioner bigPartitioner;
    private Partitioner smallPartitioner;

    @BeforeEach
    void setUp() {
        Random random = new Random(42);
        evenPartitioner = new Partitioner(10, 5);
        evenPartitioner.setRandom(random);
        unevenPartitioner = new Partitioner(10, 3);
        unevenPartitioner.setRandom(random);
        bigPartitioner = new Partitioner(10, 10);
        bigPartitioner.setRandom(random);
        smallPartitioner = new Partitioner(10, 1);
        smallPartitioner.setRandom(random);
    }

    @Test
    void constructErroneous() {
        Exception exception1 = assertThrows(IllegalArgumentException.class,
                () -> new Partitioner(1, 0));
        assertEquals("numInts (1) and setSize (0) must be positive integers.",
                exception1.getMessage());
        Exception exception2 = assertThrows(IllegalArgumentException.class,
                () -> new Partitioner(-2, 9));
        assertEquals("numInts (-2) and setSize (9) must be positive integers.",
                exception2.getMessage());
        Exception exception3 = assertThrows(IllegalArgumentException.class,
                () -> new Partitioner(4, 5));
        assertEquals("setSize (5) cannot be greater than numInts (4).",
                exception3.getMessage());
    }

    @Test
    void callEvenPartitioner() {
            assertIterableEquals(List.of(
                            List.of(4, 6, 2, 1, 7), List.of(9, 8, 5, 3, 0)),
                    evenPartitioner.call());
    }

    @Test
    void callUnevenPartitioner() {
        assertIterableEquals(List.of(
                        List.of(4, 6, 2), List.of(1, 7, 9), List.of(8, 5, 3), List.of(0)),
                unevenPartitioner.call());
    }

    @Test
    void callBigPartitioner() {
        assertIterableEquals(List.of(
                        List.of(4, 6, 2, 1, 7, 9, 8, 5, 3, 0)),
                bigPartitioner.call());
    }

    @Test
    void callSmallPartitioner() {
        assertIterableEquals(List.of(
                        List.of(4), List.of(6), List.of(2), List.of(1),
                List.of(7), List.of(9), List.of(8), List.of(5),
                List.of(3), List.of(0)),
                smallPartitioner.call());
    }
}
