package neural_network.data_generators;

import neural_network.util.Header;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public abstract class DataGeneratorTest {

    public abstract Map<String, List<Object>> getGeneratorWithExpectedDf();

    public static int erroneousClassifier() {
        return 4;
    }

    public static int oneCoordClassifier(double x1) {
        if (x1 < 0) {
            return 3;
        } else if (x1 < 1) {
            return 4;
        } else {
            return 5;
        }
    }

    public static double twoCoordRegressor(double x1, double x2) {
        return Math.round((x1 + x2) * 100.00) / 100.00;
    }

    public static String stringClassifier(double x1, double x2) {
        int intVal = (int) (x1 + x2);
        if (intVal % 2 == 0) {
            return "Even";
        } else {
            return "Odd";
        }
    }

    @SuppressWarnings("unchecked")
    @Test
    void call() throws InvocationTargetException, IllegalAccessException {
        // The below code allows for testing of multiple different generators
        // with different parameters
        Map<String, List<Object>> callMap = getGeneratorWithExpectedDf();
        int n = callMap.get("generators").size();
        for (int i = 0; i < n; i ++) {
            DataGenerator<?> generator = (DataGenerator<?>) callMap.get("generators").get(i);
            Random mockRandom = (Random) callMap.get("randoms").get(i);
            NavigableMap<Header, List<String>> expectedMap =
                    (NavigableMap<Header, List<String>>) callMap.get("dfs").get(i);
            DataGenerator.setRandom(mockRandom);
            NavigableMap<Header, List<String>> actualMap = generator.call();
            for (Header header : expectedMap.keySet()) {
                assertTrue(actualMap.containsKey(header));
                assertIterableEquals(expectedMap.get(header),
                        actualMap.get(header));
            }
        }
    }
}
