package neural_network.functions;

import org.junit.jupiter.api.BeforeEach;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SoftmaxTest extends ActivatorTest<Double> {

    private Softmax softmax;
    private final List<Double> zList = new ArrayList<>(List.of(-2.0, 0.0, 2.0, 4.0));

    @BeforeEach
    void setUp() {
        softmax = new Softmax();
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedValue() {
        // Try with initial values first and then for softmax2 use the normalise function
        Softmax softmax2 = new Softmax();
        softmax2.normalise(zList);

        Map<String, List<Object>> valueMap = new HashMap<>();
        valueMap.put("activators", List.of(softmax, softmax2, softmax2, softmax2, softmax2));
        valueMap.put("xs", List.of(0.0, zList.get(0), zList.get(1), zList.get(2), zList.get(3)));
        valueMap.put("values", List.of(1.0, 0.00214401, 0.01584220, 0.11705891, 0.86495488));
        return valueMap;
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedGradient() {
        Map<String, List<Object>> gradientMap = new HashMap<>();
        gradientMap.put("activators", List.of(softmax));
        gradientMap.put("xs", List.of(0.0));
        gradientMap.put("gradients", null);
        return gradientMap;
    }
}
