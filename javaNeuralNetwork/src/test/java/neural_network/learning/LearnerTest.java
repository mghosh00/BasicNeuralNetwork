package neural_network.learning;

import neural_network.components.Network;
import neural_network.functions.CrossEntropyLoss;
import neural_network.functions.MSELoss;
import neural_network.util.Header;
import neural_network.util.Partitioner;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.NavigableMap;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

public abstract class LearnerTest {

    public abstract Map<String, List<Object>> getLearnerWithExpectedAttributes();
    public abstract Map<String, List<Object>> getLearnerWithExpectedLoss();
    public abstract  Map<String, List<Object>> getLearnerWithExpectedCategoricalDf();

    @SuppressWarnings("unchecked")
    @Test
    void construct() {
        Map<String, List<Object>> constructMap = getLearnerWithExpectedAttributes();
        int n = constructMap.get("learners").size();
        for (int i = 0; i < n; i ++) {
            Learner learner = (Learner) constructMap.get("learners").get(i);
            Network network = (Network) constructMap.get("networks").get(i);
            boolean regressor = (boolean) constructMap.get("regressors").get(i);
            int numDatapoints = (int) constructMap.get("numDatapoints").get(i);
            int batchSize = (int) constructMap.get("batchSizes").get(i);
            List<String> categoryNames = (List<String>) constructMap.get("categoryNames").get(i);
            NavigableMap<Header, List<String>> categoricalDf =
                    (NavigableMap<Header, List<String>>) constructMap.get("categoricalDfs").get(i);
            NavigableMap<Header, List<Double>> df =
                    (NavigableMap<Header, List<Double>>) constructMap.get("dfs").get(i);
            Partitioner partitioner = (Partitioner) constructMap.get("partitioners").get(i);

            assertEquals(network, learner.getNetwork());
            assertEquals(regressor, learner.isRegressor());
            assertEquals(numDatapoints, learner.getNumDatapoints());
            assertEquals(batchSize, learner.getBatchSize());
            if (! regressor) {
                assertIterableEquals(categoryNames, learner.getCategoryNames());
            }
            assertEquals(partitioner, learner.getPartitioner());

            NavigableMap<Header, List<String>> trueCategoricalDf = learner.getCategoricalDf();
            NavigableMap<Header, List<Double>> trueDf = learner.getDf();
            for (Header header : df.keySet()) {
                if (! regressor) {
                    assertTrue(trueCategoricalDf.containsKey(header));
                    assertIterableEquals(categoricalDf.get(header),
                            trueCategoricalDf.get(header));
                }
                assertTrue(trueDf.containsKey(header));
                assertIterableEquals(df.get(header),
                        trueDf.get(header));
            }
        }
    }

    @SuppressWarnings("unchecked")
    @Test
    void forwardPassOneDatapoint() {
        Map<String, List<Object>> forwardPassMap = getLearnerWithExpectedLoss();
        int n = forwardPassMap.get("learners").size();
        for (int i = 0; i < n; i ++) {

            // We assume here that learner is a spy and network, loss have been
            // mocked
            Learner spyLearner = (Learner) forwardPassMap.get("learners").get(i);
            boolean doRegression = spyLearner.isRegressor();
            Network mockNetwork = spyLearner.getNetwork();
            NavigableMap<Header, List<Double>> df = spyLearner.getDf();

            // These are the chosen batch ids and values from the output neurons -
            // either softmax probabilities or regression values
            List<Integer> batchIds =
                    (List<Integer>) forwardPassMap.get("batchIds").get(i);
            List<List<Double>> outputNeuronVals =
                    (List<List<Double>>) forwardPassMap.get("outputNeuronVals").get(i);
            double totalLoss = (double) forwardPassMap.get("totalLosses").get(i);
            if (doRegression) {
                MSELoss mockMSELoss =
                        (MSELoss) forwardPassMap.get("lossFunctions").get(i);
                List<Double> trueYHats =
                        (List<Double>) forwardPassMap.get("yHats").get(i);

                // Assertions start here
                double trueLoss = spyLearner.forwardPassOneBatch(batchIds);
                assertEquals(totalLoss, trueLoss);
                for (int j = 0; j < batchIds.size(); j ++) {
                    int batchId = batchIds.get(j);
                    // Check that the output neuron regression value has
                    // been written to the df
                    double predictedYHat = outputNeuronVals.get(j).get(0);
                    assertEquals(predictedYHat,
                            df.get(Header.Y_HAT).get(batchId));
                    verify(mockMSELoss, times(1))
                            .call(predictedYHat, trueYHats.get(j));
                    verify(spyLearner, times(1))
                            .storeGradients(batchId);
                }
            } else {
                CrossEntropyLoss mockCrossEntropyLoss =
                        (CrossEntropyLoss) forwardPassMap.get("lossFunctions").get(i);
                List<Integer> trueYHats =
                        (List<Integer>) forwardPassMap.get("yHats").get(i);
                List<Integer> predictedYHats =
                        (List<Integer>) forwardPassMap.get("predictedYHats").get(i);

                // Assertions start here
                double trueLoss = spyLearner.forwardPassOneBatch(batchIds);
                assertEquals(totalLoss, trueLoss);
                for (int j = 0; j < batchIds.size(); j ++) {
                    int batchId = batchIds.get(j);
                    // Check that the most likely class has been written
                    // to the df
                    assertEquals(predictedYHats.get(j),
                            (int) (double) df.get(Header.Y_HAT).get(batchId));
                    verify(mockCrossEntropyLoss, times(1))
                            .call(outputNeuronVals.get(j), trueYHats.get(j));
                    verify(spyLearner, times(1))
                            .storeGradients(batchId);
                }
            }

            verify(mockNetwork, times(2))
                    .forwardPassOneDatapoint(anyList());

        }
    }

    @SuppressWarnings("unchecked")
    @Test
    void updateCategoricalDataframe() {
        Map<String, List<Object>> categoricalDfMap =
                getLearnerWithExpectedCategoricalDf();
        int n = categoricalDfMap.get("learners").size();
        for (int i = 0; i < n; i ++) {
            Learner learner = (Learner) categoricalDfMap.get("learners").get(i);
            boolean doRegression = learner.isRegressor();
            if (doRegression) {

                // If this is a regression network, we throw an exception
                // when this method is called
                Exception exception = assertThrows(RuntimeException.class,
                        learner::updateCategoricalDataframe);
                assertEquals("Cannot call updateCategoricalDataframe " +
                        "with a regression network", exception.getMessage());
            } else {

                // Otherwise we simply check that the categorical df has
                // been updated as we expect it to
                NavigableMap<Header, List<String>> categoricalDf =
                        (NavigableMap<Header, List<String>>) categoricalDfMap
                                .get("dfs").get(i);
                learner.updateCategoricalDataframe();
                NavigableMap<Header, List<String>> trueDf = learner.getCategoricalDf();
                for (Header header : categoricalDf.keySet()) {
                    assertTrue(trueDf.containsKey(header));
                    assertIterableEquals(categoricalDf.get(header),
                            trueDf.get(header));
                }
            }
        }
    }
}
