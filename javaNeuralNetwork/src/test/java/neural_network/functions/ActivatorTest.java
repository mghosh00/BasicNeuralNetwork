package neural_network.functions;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

// This class uses the template method pattern to test for any implementer
// of the Activator interface
public abstract class ActivatorTest<T> {

    public abstract Map<String, List<Object>> getActivatorWithExpectedValue();

    public abstract Map<String, List<Object>> getActivatorWithExpectedGradient();

    @SuppressWarnings("unchecked")
    @Test
    void call() {
        // The below code allows for testing of multiple different activators
        // with different parameters
        Map<String, List<Object>> callMap = getActivatorWithExpectedValue();
        int n = callMap.get("activators").size();
        for (int i = 0; i < n; i ++) {
            Activator<T> activator = (Activator<T>) callMap.get("activators").get(i);
            T x = (T) callMap.get("xs").get(i);
            double expectedValue = (double) callMap.get("values").get(i);
            assertEquals(expectedValue, activator.call(x), .00000001);
        }
    }

    @SuppressWarnings("unchecked")
    @Test
    void gradient() {
        // The below code allows for testing of multiple different activators
        // with different parameters
        Map<String, List<Object>> gradientMap = getActivatorWithExpectedGradient();

        // If gradients is null, then the gradient method should not be called,
        // and so we check that the correct exception has been thrown
        if (gradientMap.get("gradients") == null) {
            Activator<T> activator = (Activator<T>) gradientMap.get("activators").get(0);
            T x = (T) gradientMap.get("xs").get(0);
            Exception exception = assertThrows(UnsupportedOperationException.class,
                    () -> activator.gradient(x));
            assertEquals("gradient method should not be" +
                    "called from the Softmax class.", exception.getMessage());
            return;
        }

        int n = gradientMap.get("activators").size();
        for (int i = 0; i < n; i ++) {
            Activator<T> activator = (Activator<T>) gradientMap.get("activators").get(i);
            T x = (T) gradientMap.get("xs").get(i);
            T expectedGradient = (T) gradientMap.get("gradients").get(i);
            if (expectedGradient instanceof Double) {
                double expected = (double) expectedGradient;
                double actual = (double) activator.gradient(x);
                assertEquals(expected, actual, .00000001);
            } else {
                assertEquals(expectedGradient, activator.gradient(x));
            }
        }
    }
}
