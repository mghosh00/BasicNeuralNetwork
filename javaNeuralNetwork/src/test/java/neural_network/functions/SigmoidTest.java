package neural_network.functions;

import org.junit.jupiter.api.BeforeEach;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SigmoidTest extends ActivatorTest<Double> {

    private Sigmoid sigmoid;

    @BeforeEach
    void setUp() {
        this.sigmoid = new Sigmoid();
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedValue() {
        Map<String, List<Object>> valueMap = new HashMap<>();
        valueMap.put("activators", List.of(sigmoid, sigmoid, sigmoid));
        valueMap.put("xs", List.of(0.0, 1.0, -1.0));
        valueMap.put("values", List.of(0.5, 0.73105858, 0.26894142));
        return valueMap;
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedGradient() {
        Map<String, List<Object>> gradientMap = new HashMap<>();
        gradientMap.put("activators", List.of(sigmoid, sigmoid, sigmoid));
        gradientMap.put("xs", List.of(0.0, 1.0, -1.0));
        gradientMap.put("gradients", List.of(0.25, 0.19661193, 0.19661193));
        return gradientMap;
    }
}
