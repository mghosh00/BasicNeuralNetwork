package neural_network.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.XYStyler;

import java.awt.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;

public class PlotterTest {

    private final NavigableMap<Header, List<String>> scatterDf = new TreeMap<>(Map.of(
            Header.X_1, List.of("-2.0", "2.0", "-8.0", "-8.0", "2.0"),
            Header.X_2, List.of("0.0", "6.0", "-2.0", "4.0", "6.0"),
            Header.Y, List.of("neg", "pos", "neg", "neg", "pos"),
            Header.Y_HAT, List.of("neg", "neg", "pos", "neg", "pos")));
    private final NavigableMap<Header, List<Double>> regScatterDf = new TreeMap<>(Map.of(
            Header.X_1, List.of(-2.0, 2.0, -8.0, -8.0, 2.0),
            Header.X_2, List.of(0.0, 6.0, -2.0, 4.0, 6.0),
            Header.Y, List.of(0.0, 6.0, -9.0, 4.0, 6.0),
            Header.Y_HAT, List.of(0.3, 5.7, -4.2, 4.1, 5.5)));
    private final Map<String, List<Double>> lossDf = new HashMap<>(Map.of(
            "Training", List.of(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
            "Validation", List.of(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)));
    private String dirName;

    @BeforeEach
    void setUp() {
        if (Files.exists(Path.of("javaNeuralNetwork"))) {
            dirName = "javaNeuralNetwork/src/test/resources/util/plots/";
        } else {
            dirName = "src/test/resources/util/plots/";
        }
        Plotter.setDirName(dirName);
        Plotter.setShowPlots(false);
    }

    @Test
    void construct() {
        assertNull(Plotter.getChart());
        assertNull(Plotter.getWrappedChart());
        assertInstanceOf(Plotter.class, new Plotter());
    }

    @Test
    void datapointScatterErroneous() throws IOException {
        Exception exception = assertThrows(RuntimeException.class,
                () -> Plotter.datapointScatter(scatterDf, "good_phase",
                        "bad_title/", false));
        assertEquals("Invalid title: bad_title/",
                exception.getMessage());
        Files.delete(Path.of(dirName + "good_phase"));
    }

    @Test
    void datapointScatter() throws IOException {
        Plotter.setShowPlots(true);
        XYChart mockChart = mock(XYChart.class);
        XYSeries mockSeries1 = mock(XYSeries.class);
        XYSeries mockSeries2 = mock(XYSeries.class);
        when(mockChart.getHeight()).thenReturn(1);
        when(mockChart.getWidth()).thenReturn(1);
        when(mockChart.addSeries(any(String.class), anyList(), anyList()))
                .thenReturn(mockSeries1, mockSeries2);
        XYStyler mockStyler = mock(XYStyler.class);
        when(mockChart.getStyler())
                .thenReturn(mockStyler);
        SwingWrapper<XYChart> mockWrappedChart = mock(SwingWrapper.class);
        Plotter.setChart(mockChart);
        Plotter.setWrappedChart(mockWrappedChart);
        Plotter.datapointScatter(scatterDf, "validation", "test_title", false);
        verify(mockChart, times(1))
                .setTitle("Predicted classes for validation data");
        verify(mockChart, times(1))
                .getStyler();
        verify(mockStyler, times(1))
                .setMarkerSize(5);
        verify(mockChart, times(1))
                .addSeries("neg", List.of(-2.0, 2.0, -8.0), List.of(0.0, 6.0, 4.0));
        verify(mockChart, times(1))
                .addSeries("pos", List.of(-8.0, 2.0), List.of(-2.0, 6.0));
        verify(mockSeries1, times(1))
                .setMarkerColor(Color.RED);
        verify(mockSeries1, times(1))
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        verify(mockSeries2, times(1))
                .setMarkerColor(Color.BLUE);
        verify(mockSeries2, times(1))
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        verify(mockWrappedChart, times(1))
                .displayChart();
        assertNull(Plotter.getChart());
        assertNull(Plotter.getWrappedChart());
    }

    @Test
    void datapointScatterNoMocks() throws IOException {
        Plotter.datapointScatter(scatterDf, "true", "", true);
        assertNull(Plotter.getChart());
        assertNull(Plotter.getWrappedChart());
    }

    @Test
    void organiseDataByCategory() {
        NavigableMap<String, List<List<Double>>> trueMap =
                Plotter.organiseDataByCategory(scatterDf, Header.Y_HAT);
        NavigableMap<String, List<List<Double>>> expectedMap = new TreeMap<>(Map.of(
                "neg", List.of(List.of(-2.0, 2.0, -8.0), List.of(0.0, 6.0, 4.0)),
                "pos", List.of(List.of(-8.0, 2.0), List.of(-2.0, 6.0))
        ));
        for (String category : expectedMap.keySet()) {
            assertTrue(trueMap.containsKey(category));
            assertIterableEquals(trueMap.get(category), expectedMap.get(category));
        }
    }

    @Test
    void comparisonScatterErroneous() throws IOException {
        Exception exception = assertThrows(RuntimeException.class,
                () -> Plotter.comparisonScatter(regScatterDf, "good_phase",
                        "bad_title/"));
        assertEquals("Invalid title: bad_title/",
                exception.getMessage());
        Files.delete(Path.of(dirName + "good_phase"));
    }

    @Test
    void comparisonScatter() throws IOException {
        Plotter.setShowPlots(true);
        XYChart mockChart = mock(XYChart.class);
        XYSeries mockSeries1 = mock(XYSeries.class);
        XYSeries mockSeries2 = mock(XYSeries.class);
        when(mockChart.getHeight()).thenReturn(1);
        when(mockChart.getWidth()).thenReturn(1);
        when(mockChart.addSeries(any(String.class), anyList(), anyList()))
                .thenReturn(mockSeries1, mockSeries2);
        XYStyler mockStyler = mock(XYStyler.class);
        when(mockChart.getStyler())
                .thenReturn(mockStyler);
        SwingWrapper<XYChart> mockWrappedChart = mock(SwingWrapper.class);
        Plotter.setChart(mockChart);
        Plotter.setWrappedChart(mockWrappedChart);
        Plotter.comparisonScatter(regScatterDf, "training", "test_title");
        verify(mockChart, times(1))
                .setTitle("Comparison scatter plot for training data");
        verify(mockChart, times(1))
                .getStyler();
        verify(mockStyler, times(1))
                .setMarkerSize(5);
        verify(mockChart, times(1))
                .addSeries("y = x", List.of(-10.0, 7.0), List.of(-10.0, 7.0));
        verify(mockChart, times(1))
                .addSeries("Regression values",
                        List.of(0.0, 6.0, -9.0, 4.0, 6.0), List.of(0.3, 5.7, -4.2, 4.1, 5.5));
        verify(mockSeries1, times(1))
                .setMarkerColor(Color.RED);
        verify(mockSeries1, times(1))
                .setLineColor(Color.RED);
        verify(mockSeries1, times(1))
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        verify(mockSeries2, times(1))
                .setMarkerColor(Color.BLACK);
        verify(mockSeries2, times(1))
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        verify(mockWrappedChart, times(1))
                .displayChart();
        assertNull(Plotter.getChart());
        assertNull(Plotter.getWrappedChart());
    }

    @Test
    void comparisonScatterNoMocks() throws IOException {
        Plotter.comparisonScatter(regScatterDf, "true", "");
        assertNull(Plotter.getChart());
        assertNull(Plotter.getWrappedChart());
    }


    @Test
    void plotLossErroneous() throws IOException {
        Exception exception = assertThrows(RuntimeException.class,
                () -> Plotter.plotLoss(lossDf,"bad_title/"));
        assertEquals("Invalid title: bad_title/",
                exception.getMessage());
    }

    @Test
    void plotLoss() throws IOException {
        Plotter.setShowPlots(true);
        XYChart mockChart = mock(XYChart.class);
        XYSeries mockSeries1 = mock(XYSeries.class);
        XYSeries mockSeries2 = mock(XYSeries.class);
        when(mockChart.getHeight()).thenReturn(1);
        when(mockChart.getWidth()).thenReturn(1);
        when(mockChart.addSeries(any(String.class), anyList()))
                .thenReturn(mockSeries1, mockSeries2);
        XYStyler mockStyler = mock(XYStyler.class);
        when(mockChart.getStyler())
                .thenReturn(mockStyler);
        SwingWrapper<XYChart> mockWrappedChart = mock(SwingWrapper.class);
        Plotter.setChart(mockChart);
        Plotter.setWrappedChart(mockWrappedChart);
        Plotter.plotLoss(lossDf, "test_title");
        verify(mockChart, times(1))
                .getStyler();
        verify(mockStyler, times(1))
                .setMarkerSize(1);
        verify(mockChart, times(1))
                .addSeries("Training",
                        List.of(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1));
        verify(mockChart, times(1))
                .addSeries("Validation",
                        List.of(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2));
        verify(mockSeries1, times(1))
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        verify(mockSeries2, times(1))
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        verify(mockWrappedChart, times(1))
                .displayChart();
        assertNull(Plotter.getChart());
        assertNull(Plotter.getWrappedChart());
    }

    @Test
    void plotLossNoMocks() throws IOException {
        Plotter.plotLoss(lossDf, "");
        assertNull(Plotter.getChart());
        assertNull(Plotter.getWrappedChart());
    }
}
