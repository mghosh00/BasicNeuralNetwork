package neural_network.util;

import org.knowm.xchart.*;

import java.awt.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.List;

/** Class to create plots evaluating the performance of the neural network.
 * This class uses the org.knowm.xchart library to create the plots.
 *
 */
public class Plotter {

    private static String dirName = "plots/";
    private static boolean showPlots = false;
    private static final List<Color> colours = List.of(Color.RED, Color.BLUE, Color.MAGENTA,
            Color.GREEN, Color.YELLOW, Color.ORANGE, Color.CYAN, Color.PINK);
    private static XYChart chart = null;
    private static SwingWrapper<XYChart> wrappedChart = null;

    /** Creates a scatter plot of the predicted/true classes for a given set
     * of data.
     *
     * @param df The data containing datapoints with true or predicted values.
     * @param phase The phase of learning (training/validation/testing) or "true"
     *              for the true data.
     * @param title The title of the plot.
     * @param regression Whether this is for regression or classification data.
     */
    public static void datapointScatter(NavigableMap<Header, List<String>> df, String phase,
                                        String title, boolean regression) throws IOException {
        // Create new directory if it does not exist
        Path path = Path.of(dirName + phase);
        if (! (Files.exists(path))) {
            Files.createDirectories(path);
        }
        Header yHeader = (phase.equals("true")) ? Header.Y : Header.Y_HAT;
        String actualOrPredicted = (phase.equals("true")) ? "Actual" : "Predicted";
        String valuesOrClasses = (regression) ? "values" : "classes";
        String subString = (title.isEmpty()) ? "" : "_" + title;
        // Create chart instance if it has not been set already
        if (chart == null) {
            chart = new XYChartBuilder().width(800).height(600)
                    .xAxisTitle("x1").yAxisTitle("x2").build();
        }
        chart.setTitle("%s %s for %s data".formatted(actualOrPredicted, valuesOrClasses, phase));
        chart.getStyler().setMarkerSize(5);
        if (! regression) {
            NavigableMap<String, List<List<Double>>> dataByCategory = organiseDataByCategory(
                    df, yHeader);
            int colourIndex = 0;
            for (String category : dataByCategory.keySet()) {
                List<Double> x1 = dataByCategory.get(category).get(0);
                List<Double> x2 = dataByCategory.get(category).get(1);
                XYSeries series = chart.addSeries(category, x1, x2);
                series.setMarkerColor(colours.get(colourIndex));
                series.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
                colourIndex ++;
            }
        }

        if (wrappedChart == null) {
            wrappedChart = new SwingWrapper<>(chart);
        }
        // Potentially show the chart and then save it
        if (showPlots) {
            wrappedChart.displayChart();
        }
        try {
            BitmapEncoder.saveBitmap(chart, dirName + phase + "/scatter" + subString,
                    BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            throw new RuntimeException("Invalid title: " + title);
        }

        // Reset the chart and wrappedChart
        chart = null;
        wrappedChart = null;
    }

    /** Takes in a categorical df from the {@code Learning} phases and then creates
     * a new df keyed by the category names, with x1 and x2 data inside.
     *
     * @param df Categorical frame returned by a {@code Learner}.
     * @param yHeader Whether we look for true ({@code Y}) or predicted ({@code Y_HAT}) classes.
     * @return The new dataframe.
     */
    static NavigableMap<String, List<List<Double>>> organiseDataByCategory(
            NavigableMap<Header, List<String>> df, Header yHeader) {
        NavigableMap<String, List<List<Double>>> dataByCategory = new TreeMap<>();
        for (int i = 0; i < df.get(yHeader).size(); i++) {
            String category = df.get(yHeader).get(i);
            double x1 = Double.parseDouble(df.get(Header.X_1).get(i));
            double x2 = Double.parseDouble(df.get(Header.X_2).get(i));
            List<List<Double>> catData = dataByCategory.computeIfAbsent(category, k -> new ArrayList<>(
                    List.of(new ArrayList<>(), new ArrayList<>())));
            catData.get(0).add(x1);
            catData.get(1).add(x2);
        }
        return dataByCategory;
    }

    /** Creates a scatter plot comparing the true and predicted values from
     * the network. This method is for regression only. The classification equivalent
     * is {@code tester.generateConfusion()}.
     *
     * @param df The numerical dataframe from a regression problem.
     * @param phase The phase of learning (training/validation/testing).
     * @param title The title of the plot.
     */
    public static void comparisonScatter(NavigableMap<Header, List<Double>> df,
                                         String phase, String title) throws IOException {
        // Create new directory if it does not exist
        Path path = Path.of(dirName + phase);
        if (! (Files.exists(path))) {
            Files.createDirectories(path);
        }
        String subString = (title.isEmpty()) ? "" : "_" + title;
        // Create chart instance if it is null
        if (chart == null) {
            chart = new XYChartBuilder().width(800).height(600)
                    .xAxisTitle("Actual").yAxisTitle("Predicted").build();
        }
        chart.setTitle("Comparison scatter plot for %s data".formatted(phase));
        chart.getStyler().setMarkerSize(5);

        // Add the line y = x as we want to get as close to this line as possible
        double minY = Collections.min(df.get(Header.Y)); double maxY = Collections.max(df.get(Header.Y));
        double delta = (maxY - minY) / 15;
        XYSeries line = chart.addSeries("y = x", List.of(minY - delta, maxY + delta),
                List.of(minY - delta, maxY + delta));
        line.setLineColor(Color.RED);
        line.setMarkerColor(Color.RED);
        line.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);

        // Add the comparison scatter
        XYSeries series = chart.addSeries("Regression values", df.get(Header.Y),
                df.get(Header.Y_HAT));
        series.setMarkerColor(Color.BLACK);
        series.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);

        // Potentially show the chart and then save it
        if (wrappedChart == null) {
            wrappedChart = new SwingWrapper<>(chart);
        }
        if (showPlots) {
            wrappedChart.displayChart();
        }
        try {
            BitmapEncoder.saveBitmap(chart, dirName + phase + "/comparison" + subString,
                    BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            throw new RuntimeException("Invalid title: " + title);
        }

        // Reset the chart and wrappedChart
        chart = null;
        wrappedChart = null;
    }

    /** Plots (normally) the training and validation losses over time.
     *
     * @param lossDf The loss data.
     * @param title A title for the plot.
     */
    public static void plotLoss(Map<String, List<Double>> lossDf, String title) throws IOException {
        // Create new directory if it does not exist
        Path path = Path.of(dirName);
        if (! (Files.exists(path))) {
            Files.createDirectories(path);
        }
        String subString = (title.isEmpty()) ? "" : "_" + title;
        // Create chart instance if it is null
        if (chart == null) {
            chart = new XYChartBuilder().width(800).height(600)
                    .title("Loss over time")
                    .xAxisTitle("Epoch").yAxisTitle("Loss").build();
        }
        chart.getStyler().setMarkerSize(1);

        // Add the different loss traces from the df
        for (String phase : lossDf.keySet()) {
            XYSeries line = chart.addSeries(phase, lossDf.get(phase));
            line.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        }

        // Potentially show the chart and then save it
        if (wrappedChart == null) {
            wrappedChart = new SwingWrapper<>(chart);
        }
        if (showPlots) {
            wrappedChart.displayChart();
        }
        try {
            BitmapEncoder.saveBitmap(chart, dirName + "/losses" + subString,
                    BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            throw new RuntimeException("Invalid title: " + title);
        }

        chart = null;
        wrappedChart = null;
    }

    /** Setter for dirName.
     *
     * @param dirName The overall directory for all plots to reside in.
     */
    public static void setDirName(String dirName) {
        Plotter.dirName = dirName;
    }

    /** Setter for showPlots.
     *
     * @param showPlots Whether we show plots on execution or not.
     */
    public static void setShowPlots(boolean showPlots) {
        Plotter.showPlots = showPlots;
    }

    /** Setter for chart. Used for mocking.
     *
     * @param chart The chart instance from {@code org.knowm.xchart}.
     */
    static void setChart(XYChart chart) {
        Plotter.chart = chart;
    }

    /** Setter for wrapperChart. Used for mocking.
     *
     * @param wrappedChart The swing wrapper instance from {@code org.knowm.xchart}.
     */
    static void setWrappedChart(SwingWrapper<XYChart> wrappedChart) {
        Plotter.wrappedChart = wrappedChart;
    }

    /** Setter for chart. Used for mocking.
     *
     * @return The chart instance from {@code org.knowm.xchart}.
     */
    static XYChart getChart() {
        return chart;
    }

    /** Setter for wrappedChart. Used for mocking.
     *
     * @return The wrappedChart instance from {@code org.knowm.xchart}.
     */
    static SwingWrapper<XYChart> getWrappedChart() {
        return wrappedChart;
    }
}
