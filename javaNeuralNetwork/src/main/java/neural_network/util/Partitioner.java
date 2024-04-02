package neural_network.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/** Class to randomly partition {@code numInts} integers into sets of size
 * {@code setSize}.
 *
 */
public class Partitioner {

    private final int numInts;
    private final int setSize;
    private Random random = new Random();

    /** Constructor method.
     *
     * @param numInts Number of integers.
     * @param setSize Size of each set.
     */
    public Partitioner(int numInts, int setSize) {
        if (numInts <= 0 || setSize <= 0) {
            throw new IllegalArgumentException(
                    "numInts (%d) and setSize (%d) must be positive integers."
                            .formatted(numInts, setSize));
        } else if (setSize > numInts) {
            throw new IllegalArgumentException(
                    "setSize (%d) cannot be greater than numInts (%d)."
                            .formatted(setSize, numInts));
        }
        this.numInts = numInts;
        this.setSize = setSize;
    }

    /** Shuffles all integers from {@code 0} to {@code numInts - 1} and creates
     * a partition of this list.
     *
     * @return The partitioned list.
     */
    public List<List<Integer>> call() {
        List<Integer> ints = new ArrayList<>(IntStream.range(0, numInts)
                .boxed().toList());
        Collections.shuffle(ints, random);
        List<List<Integer>> outputList = new ArrayList<>();
        int numSets = (int) Math.ceil((double) numInts / setSize);
        for (int i = 0; i < numSets - 1; i ++) {
            outputList.add(ints.subList(i * setSize, (i + 1) * setSize));
        }
        outputList.add(ints.subList((numSets - 1) * setSize, numInts));
        return outputList;
    }

    public void setRandom(Random random) {
        this.random = random;
    }
}
