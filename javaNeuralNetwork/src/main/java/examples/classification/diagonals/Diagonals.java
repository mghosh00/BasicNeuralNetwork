package examples.classification.diagonals;

import neural_network.components.Network;
import neural_network.data_generators.DataGenerator;
import neural_network.data_generators.NormalDataGenerator;
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

public class Diagonals {

    private static final String resourcesPath =
            "javaNeuralNetwork/src/main/resources/examples/classification/diagonals/";

    public static void main(String[] args) throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        Plotter.setDirName(resourcesPath + "plots/");
        generateData();
        run();
    }

    public static void generateData() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException, IOException {
        Method classifier = Diagonals.class.getDeclaredMethod("classifier",
                double.class, double.class);
        DataGenerator<String> generator = new NormalDataGenerator<>(classifier, 400,
                List.of(0.0, 0.0), List.of(1.0, 1.0));
        generator.call();
        generator.plotDatapoints("diagonals", false);
        generator.writeToCsv("diagonals_data", resourcesPath);
    }

    /** This classifies points according to diagonals x_1 = x_2 and x_1 = -x_2.
     *
     * @param x1 Horizontal coordinate.
     * @param x2 Vertical coordinate.
     * @return Positional string (north, south, east, west).
     */
    public static String classifier(double x1, double x2) {
        if (x1 + x2 > 0) {
            return (x1 - x2 > 0) ? "East" : "North";
        }
        return (x1 - x2 > 0) ? "South" : "West";
    }

    public static void run() throws IOException {
        // Split the data
        DataSplitter splitter = new DataSplitter(resourcesPath + "diagonals_data.csv",
                List.of(8, 1, 1));
        List<NavigableMap<Header, List<String>>> data = splitter.split();
        NavigableMap<Header, List<String>> trainingData = data.get(0);
        NavigableMap<Header, List<String>> validationData = data.get(1);
        NavigableMap<Header, List<String>> testingData = data.get(2);

        // Set up the network
        double learningRate = 0.01;
        Network network = new Network(2, 2, List.of(4, 4),
                4, 0.01, learningRate, false, true, 0.9, true);

        // Create different phases of learning
        Validator validator = new Validator(network, validationData, 10);
        Trainer trainer = new Trainer(network, trainingData, 16, true, 10,
                1000, validator);
        Tester tester = new Tester(network, testingData, 10);

        // Run the learning and generate plots
        trainer.run();
        Plotter.setShowPlots(true);
        trainer.generateLossPlot("diagonals");
        trainer.generateScatter("diagonals");
        validator.generateScatter("diagonals");
        tester.run();
        tester.generateScatter("diagonals");
        tester.generateConfusion();
    }
}
