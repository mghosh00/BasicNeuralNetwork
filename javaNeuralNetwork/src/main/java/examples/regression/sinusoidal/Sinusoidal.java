package examples.regression.sinusoidal;

import neural_network.components.Network;
import neural_network.data_generators.DataGenerator;
import neural_network.data_generators.UniformDataGenerator;
import neural_network.learning.Tester;
import neural_network.learning.Trainer;
import neural_network.learning.Validator;
import neural_network.util.DataSplitter;
import neural_network.util.Header;
import neural_network.util.Plotter;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.NavigableMap;

public class Sinusoidal {

    private static final String resourcesPath =
            "javaNeuralNetwork/src/main/resources/examples/regression/sinusoidal/";

    public static void main(String[] args) throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        Plotter.setDirName(resourcesPath + "plots/");
        generateData();
        run();
    }

    public static void generateData() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException, IOException {
        Method regressor = Sinusoidal.class.getDeclaredMethod("sineRegressor",
                double.class, double.class);
        DataGenerator<String> generator = new UniformDataGenerator<>(regressor, 800,
                List.of(-3.14, -3.14), List.of(3.14, 3.14));
        generator.call();
        generator.plotDatapoints("sine", true);
        generator.writeToCsv("sinusoidal_data", resourcesPath);
    }

    /** This returns the product of the sines of each coordinate.
     *
     * @param x1 Horizontal coordinate.
     * @param x2 Vertical coordinate.
     * @return {@code sin(x1) * sin(x2)}
     */
    public static double sineRegressor(double x1, double x2) {
        return Math.sin(x1) * Math.sin(x2);
    }

    public static void run() throws IOException {
        // Split the data
        DataSplitter splitter = new DataSplitter(resourcesPath + "sinusoidal_data.csv",
                List.of(8, 1, 1));
        List<NavigableMap<Header, List<String>>> data = splitter.split();
        NavigableMap<Header, List<String>> trainingData = data.get(0);
        NavigableMap<Header, List<String>> validationData = data.get(1);
        NavigableMap<Header, List<String>> testingData = data.get(2);

        // Set up the network
        double learningRate = 0.001;
        Network network = new Network(2, 3, List.of(6, 6, 6),
                2, 0.01, learningRate, true, true, 0.9, true);

        // Create different phases of learning
        Validator validator = new Validator(network, validationData, 10);
        Trainer trainer = new Trainer(network, trainingData, 16, true, 10,
                1000, validator);
        Tester tester = new Tester(network, testingData, 10);

        // Run the learning and generate plots
        trainer.run();
        Plotter.setShowPlots(true);
        trainer.generateLossPlot("sine");
        trainer.comparisonScatter("sine");
        tester.run();
        tester.comparisonScatter("sine");
    }
}
