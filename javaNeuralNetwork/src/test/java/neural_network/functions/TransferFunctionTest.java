package neural_network.functions;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TransferFunctionTest extends ActivatorTest<List<Double>> {

    private TransferFunction transfer;
    private final List<Double> oList = new ArrayList<>(List.of(1.0, 2.0, 3.0));
    private final List<Double> wList = new ArrayList<>(List.of(-1.0, 0.0, 1.0));
    private final double bias = 2.0;

    @BeforeEach
    void setUp() {
        this.transfer = new TransferFunction();
    }

    @Test
    void callNoBindingExceptions() {
        // No binding at all
        Exception exception1 = assertThrows(IllegalStateException.class,
                () -> transfer.call(oList));
        assertEquals("weights must have same size as o," +
                "check that weights have been bound (0 != 3)",
                exception1.getMessage());

        // Just weights binding
        transfer.bindWeights(wList);
        Exception exception2 = assertThrows(IllegalStateException.class,
                () -> transfer.call(oList));
        assertEquals("bias has not yet been bound, cannot" +
                "call this method",
                exception2.getMessage());

        // Bind both but then try to use call twice in a row
        transfer.bindBias(bias);
        transfer.call(oList);
        Exception exception3 = assertThrows(IllegalStateException.class,
                () -> transfer.call(oList));
        assertEquals("weights must have same size as o," +
                        "check that weights have been bound (0 != 3)",
                exception3.getMessage());
    }

    @Test
    void callWeightsWrongSizeException() {
        transfer.bindWeights(List.of(2.0, 3.0));
        Exception exception = assertThrows(IllegalStateException.class,
                () -> transfer.call(oList));
        assertEquals("weights must have same size as o," +
                "check that weights have been bound (2 != 3)",
                exception.getMessage());
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedValue() {
        transfer.bindWeights(wList);
        transfer.bindBias(bias);

        // Check that binding more than once has the desired effect i.e.
        // the first bind does nothing and the second bind is used correctly
        TransferFunction transfer2 = new TransferFunction();
        transfer2.bindWeights(List.of(1.0, 2.0));
        transfer2.bindBias(3.0);
        transfer2.bindWeights(List.of(4.0, 5.0));
        transfer2.bindBias(1.0);

        // Use the tests from the base class
        Map<String, List<Object>> valueMap = new HashMap<>();
        valueMap.put("activators", List.of(transfer, transfer2));
        valueMap.put("xs", List.of(oList, List.of(-2.0, -3.0)));
        valueMap.put("values", List.of(4.0, -22.0));
        return valueMap;
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedGradient() {
        Map<String, List<Object>> gradientMap = new HashMap<>();
        gradientMap.put("activators", List.of(transfer));
        gradientMap.put("xs", List.of(oList));
        List<Double> gradient = new ArrayList<>(oList);
        gradient.add(0.0);
        gradientMap.put("gradients", List.of(gradient));
        return gradientMap;
    }
}
