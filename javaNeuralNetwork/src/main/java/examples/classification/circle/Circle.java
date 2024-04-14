package examples.classification.circle;

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

/** The {@code Circle} example can be run as a gradle task labelled {@code runCircle}.
 *
 */
public class Circle {

    private static final String resourcesPath = "javaNeuralNetwork/src/main/resources/examples/classification/circle";

    public static void main(String[] args) throws NoSuchMethodException, InvocationTargetException, IllegalAccessException, IOException {
        Plotter.setDirName(resourcesPath + "/plots/");
        generateData();
        run();
    }

    public static void generateData() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException, IOException {
        Method classifier = Circle.class.getDeclaredMethod("classifier",
                double.class, double.class);
        DataGenerator<String> generator = new UniformDataGenerator<>(classifier, 300,
                List.of(-1.0, -1.0), List.of(1.0, 1.0));
        generator.call();
        generator.plotDatapoints("circle", false);
        generator.writeToCsv("circle_data", resourcesPath);
    }

    /** The classifier method for the circle example.
     *
     * @param x1 The first x coordinate.
     * @param x2 The second x coordinate.
     * @return Whether the point is inside or outside the unit disc.
     */
    public static String classifier(double x1, double x2) {
        if (x1 * x1 + x2 * x2 <= 1) {
            return "Inside";
        } else {
            return "Outside";
        }
    }

    public static void run() throws IOException {
        // Split the data
        DataSplitter splitter = new DataSplitter(resourcesPath + "/circle_data.csv",
                List.of(8, 1, 1));
        List<NavigableMap<Header, List<String>>> data = splitter.split();
        NavigableMap<Header, List<String>> trainingData = data.get(0);
        NavigableMap<Header, List<String>> validationData = data.get(1);
        NavigableMap<Header, List<String>> testingData = data.get(2);

        // Set up the network
        double learningRate = 0.01;
        Network network = new Network(2, 3, List.of(4, 4, 4),
                2, 0.01, learningRate, false, true, 0.09, true);

        // Create different phases of learning
        Validator validator = new Validator(network, validationData, 10);
        Trainer trainer = new Trainer(network, trainingData, 16, true, 10,
                1000, validator);
        Tester tester = new Tester(network, testingData, 10);

        // Run the learning and generate plots
        trainer.run();
        Plotter.setShowPlots(true);
        trainer.generateLossPlot("circle");
        trainer.generateScatter("circle");
        validator.generateScatter("circle");
        tester.run();
        tester.generateScatter("circle");
        tester.generateConfusion();
    }
}
