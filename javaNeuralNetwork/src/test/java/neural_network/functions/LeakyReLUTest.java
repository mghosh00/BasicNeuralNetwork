package neural_network.functions;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LeakyReLUTest extends ActivatorTest<Double> {

    private LeakyReLU leakyReLU;
    private LeakyReLU defaultReLU;

    @BeforeEach
    void setUp() {
        this.leakyReLU = new LeakyReLU(0.01);
        this.defaultReLU = new LeakyReLU();
    }

    @Test
    void construct() {
        assertEquals(0.01, leakyReLU.getLeak());
        assertEquals(0.0, defaultReLU.getLeak());
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedValue() {
        Map<String, List<Object>> valueMap = new HashMap<>();
        valueMap.put("activators", List.of(leakyReLU, defaultReLU, leakyReLU));
        valueMap.put("xs", List.of(2.0, -3.0, -4.0));
        valueMap.put("values", List.of(2.0, -0.0, -0.04));
        return valueMap;
    }

    @Override
    public Map<String, List<Object>> getActivatorWithExpectedGradient() {
        Map<String, List<Object>> gradientMap = new HashMap<>();
        gradientMap.put("activators", List.of(leakyReLU, defaultReLU, leakyReLU, leakyReLU));
        gradientMap.put("xs", List.of(2.0, -3.0, -4.0, 0.0));
        gradientMap.put("gradients", List.of(1.0, 0.0, 0.01, 1.0));
        return gradientMap;
    }
}
