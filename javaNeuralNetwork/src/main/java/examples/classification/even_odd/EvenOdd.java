package examples.classification.even_odd;

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

public class EvenOdd {

    private static final String resourcesPath =
            "javaNeuralNetwork/src/main/resources/examples/classification/even_odd/";

    public static void main(String[] args) throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        Plotter.setDirName(resourcesPath + "plots/");
        generateData();
        run();
    }

    public static void generateData() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException, IOException {
        Method classifier = EvenOdd.class.getDeclaredMethod("classifier",
                double.class, double.class);
        DataGenerator<String> generator = new UniformDataGenerator<>(classifier, 400,
                List.of(0.0, 0.0), List.of(6.0, 6.0));
        generator.call();
        generator.plotDatapoints("even_odd", false);
        generator.writeToCsv("even_odd_data", resourcesPath);
    }

    /** This classifies points according to whether their x1 coordinate is even or odd.
     *
     * @param x1 Horizontal coordinate.
     * @param x2 Vertical coordinate.
     * @return Whether integer part of {@code x1} is even or odd.
     */
    public static String classifier(double x1, double x2) {
        return ((int) x1 % 2 == 0) ? "Even" : "Odd";
    }

    public static void run() throws IOException {
        // Split the data
        DataSplitter splitter = new DataSplitter(resourcesPath + "even_odd_data.csv",
                List.of(8, 1, 1));
        List<NavigableMap<Header, List<String>>> data = splitter.split();
        NavigableMap<Header, List<String>> trainingData = data.get(0);
        NavigableMap<Header, List<String>> validationData = data.get(1);
        NavigableMap<Header, List<String>> testingData = data.get(2);

        // Set up the network
        double learningRate = 0.01;
        Network network = new Network(2, 3, List.of(7, 7, 7),
                2, 0.01, learningRate, false, true, 0.9, true);

        // Create different phases of learning
        Validator validator = new Validator(network, validationData, 10);
        Trainer trainer = new Trainer(network, trainingData, 16, true, 10,
                8000, validator);
        Tester tester = new Tester(network, testingData, 10);

        // Run the learning and generate plots
        trainer.run();
        Plotter.setShowPlots(true);
        trainer.generateLossPlot("even_odd");
        trainer.generateScatter("even_odd");
        validator.generateScatter("even_odd");
        tester.run();
        tester.generateScatter("even_odd");
        tester.generateConfusion();
    }
}
