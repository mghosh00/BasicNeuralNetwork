package neural_network.functions;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MSELossTest {

    private MSELoss mseLoss;

    @BeforeEach
    void setUp() {
        mseLoss = new MSELoss();
    }

    @Test
    void call() {
        assertEquals(0.25, mseLoss.call(0.8, 1.3), .00000001);
        assertEquals(0.0, mseLoss.call(0.8, 0.8), .00000001);
    }

    @Test
    void gradient() {
        assertEquals(-1.0, mseLoss.gradient(0.8, 1.3), .00000001);
        assertEquals(0.0, mseLoss.gradient(0.8, 0.8), .00000001);
    }
}
