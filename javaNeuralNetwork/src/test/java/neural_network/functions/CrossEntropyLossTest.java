package neural_network.functions;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CrossEntropyLossTest {

    private final List<Double> yHat = new ArrayList<>(List.of(-1.0, 0.4, 0.2, 2.0));


    @Test
    void callErroneous() {
        Exception exception = assertThrows(IllegalArgumentException.class,
                () -> CrossEntropyLoss.call(yHat, 0));
        assertEquals("Softmax value should be between 0" +
                " and 1 (yHat.get(0) = -1.000000)", exception.getMessage());
    }

    @Test
    void call() {
        assertEquals(0.91629073, CrossEntropyLoss.call(yHat, 1),
                .00000001);
    }
}
