package neural_network.util;

import java.util.*;
import java.util.stream.Stream;

/** Class to create a number of sets of size {@code setSize} from a list of
 * {@code numInts} integers weighted by which ground truth class each integer lies in.
 *
 */
public class WeightedPartitioner extends Partitioner {

    private int numBins;
    private final Map<Integer, List<Integer>> classMap = new HashMap<>();

    /** Constructor method.
     *
     * @param numInts Number of integers.
     * @param setSize Size of each set.
     * @param yVals The classes/values for each datapoint.
     * @param doRegression Whether we have a regression or classification problem.
     * @param numBins If {@code doRegression} is {@code true}, this represents the
     *                number of bins to split the data into. Otherwise, this
     *                parameter is ignored.
     */
    public WeightedPartitioner(int numInts, int setSize, List<Double> yVals,
                               boolean doRegression, int numBins) {
        super(numInts, setSize);
        if (! (yVals.size() == numInts)) {
            throw new IllegalArgumentException("numInts must equal the length of the " +
                    "yVals (numInts = %d, yVals.size() = %d)"
                            .formatted(numInts, yVals.size()));
        }

        // For regression, we split data into bins
        if (doRegression) {
            this.numBins = numBins;
            NavigableMap<Integer, List<Integer>> initialMap = new TreeMap<>();
            Stream.iterate(0, j -> j < numBins, j -> j + 1)
                    .forEach(j -> initialMap.put(j, new ArrayList<>(List.of())));
            double minY = Collections.min(yVals); double maxY = Collections.max(yVals);
            double classWidth = (maxY - minY) / numBins;
            for (int i = 0; i < yVals.size(); i ++) {

                // Here, we add the index of the y value to a specific discrete bin,
                // depending on where it lies in the range (minY, maxY)
                double y = yVals.get(i);
                int chosenBin = (int) ((y - minY) / classWidth);
                if (chosenBin == numBins) {
                    chosenBin = numBins - 1;
                }
                initialMap.get(chosenBin).add(i);
            }

            // We also need to account for potentially empty class lists
            for (int j = 0; j < this.numBins; j ++) {
                if (initialMap.get(j).isEmpty()) {
                    // Remove the bin and reduce numBins
                    initialMap.remove(j);
                    this.numBins --;
                }
            }

            // Finally, relabel the bins to 0, 1, 2, ...
            for (int k = 0; k < this.numBins; k ++) {
                // The below line polls the first key in the initialMap and uses it
                // to retrieve the list of integers at this entry. Then, this is put
                // in the new classMap with relabelled key
                classMap.put(k, initialMap.get(initialMap.firstKey()));
                initialMap.remove(initialMap.firstKey());
            }
        } else {
            // For classification, this process just involves getting the classes
            List<Integer> yClasses = yVals.stream()
                    .map(x -> (int) (double) x)
                    .toList();
            this.numBins = (new HashSet<>(yVals)).size();
            for (int i = 0; i < yClasses.size(); i ++) {
                classMap.computeIfAbsent(yClasses.get(i),
                        k -> new ArrayList<>()).add(i);
            }
        }
    }

    /** Constructor for classification. Note that {@code numBins} will not be used
     * in a classification problem.
     *
     * @param numInts Number of integers.
     * @param setSize Size of each set.
     * @param yVals The classes/values for each datapoint.
     */
    public WeightedPartitioner(int numInts, int setSize, List<Double> yVals) {
        this(numInts, setSize, yVals, false, 10);
    }

    /** Uses weights for each class to create sets of size {@code setSize} containing
     * integers (sampled with replacement).
     *
     * @return The list of lists of datapoint indices.
     */
    public List<List<Integer>> call() {
        // This produces a list of class indices of size numInts
        List<Integer> chosenClasses = getRandom()
                .ints(getNumInts(), 0, numBins)
                .boxed().toList();

        List<List<Integer>> outputList = new ArrayList<>();
        int numSets = (int) Math.ceil((double) getNumInts() / getSetSize());
        for (int i = 0; i < numSets - 1; i ++) {
            List<Integer> innerList = new ArrayList<>();
            for (int j = 0; j < getSetSize(); j ++) {
                int chosenClass = chosenClasses.get(i * getSetSize() + j);
                List<Integer> classList = classMap.get(chosenClass);
                int chosenInt = classList.get(
                        getRandom().nextInt(classList.size()));
                innerList.add(chosenInt);
            }
            outputList.add(innerList);
        }
        List<Integer> finalList = new ArrayList<>();
        for (int k = (numSets - 1) * getSetSize(); k < getNumInts(); k ++) {
            int chosenClass = chosenClasses.get(k);
            List<Integer> classList = classMap.get(chosenClass);
            int chosenInt = classList.get(
                    getRandom().nextInt(classList.size()));
            finalList.add(chosenInt);
        }
        outputList.add(finalList);
        return outputList;
    }

    /** Getter for {@code classMap}. Mainly for testing purposes.
     *
     * @return A copy of the {@code classMap}.
     */
    Map<Integer, List<Integer>> getClassMap() {
        Map<Integer, List<Integer>> returnMap = new HashMap<>();
        for (Integer k : classMap.keySet()) {
            returnMap.put(k, List.copyOf(classMap.get(k)));
        }
        return returnMap;
    }
}
