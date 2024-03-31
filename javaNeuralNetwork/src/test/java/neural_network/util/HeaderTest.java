package neural_network.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class HeaderTest {

    @BeforeEach
    void setUp() {
        Header.setDimensions(0);
    }

    @Test
    void setDimensionsErroneous() {
        Exception exception1 = assertThrows(IllegalArgumentException.class,
                () -> Header.setDimensions(-1));
        assertEquals("Invalid dimensions passed: -1, must be between 0 and 9.",
                exception1.getMessage());
        Exception exception2 = assertThrows(IllegalArgumentException.class,
                () -> Header.setDimensions(12));
        assertEquals("Invalid dimensions passed: 12, must be between 0 and 9.",
                exception2.getMessage());
    }

    @Test
    void setDimensions() {
        Header.setDimensions(5);
        assertEquals(5, Header.getDimensions());
    }

    @Test
    void getHeader() {
        Header.setDimensions(7);
        Header header = Header.getHeader("x_6");
        assertEquals(Header.X_6, header);
    }

    @Test
    void getInitialHeadersErroneous() {
        Exception exception = assertThrows(IllegalStateException.class,
                Header::getInitialHeaders);
        assertEquals("dimensions of Header not yet initialised",
                exception.getMessage());
    }

    @Test
    void getInitialHeaders() {
        Header.setDimensions(8);
        List<Header> headers = Header.getInitialHeaders();
        assertIterableEquals(List.of(Header.X_1, Header.X_2, Header.X_3,
                Header.X_4, Header.X_5, Header.X_6, Header.X_7, Header.X_8,
                Header.Y), headers);
    }

    @Test
    void getAllHeadersErroneous() {
        Exception exception = assertThrows(IllegalStateException.class,
                Header::getAllHeaders);
        assertEquals("dimensions of Header not yet initialised",
                exception.getMessage());
    }

    @Test
    void getAllHeaders() {
        Header.setDimensions(5);
        List<Header> headers = Header.getAllHeaders();
        assertIterableEquals(List.of(Header.X_1, Header.X_2, Header.X_3,
                Header.X_4, Header.X_5, Header.Y, Header.Y_HAT), headers);
    }

    @Test
    void string() {
        assertEquals("x_3", Header.X_3.toString());
    }
}
