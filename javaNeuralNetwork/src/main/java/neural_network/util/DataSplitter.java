package neural_network.util;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

/** Class to split a dataset into training, validation and testing, given a
 * split ratio.
 *
 */
public class DataSplitter {

    private final String path;
    private final NavigableMap<Header, List<String>> df;
    private final List<Integer> proportions;
    private CSVFormat csvFormat = null;

    /** Constructor method.
     *
     * @param path Path to the .csv file containing the data.
     * @param proportions The proportions in the sequence training:validation:testing.
     */
    public DataSplitter(String path, List<Integer> proportions) {
        this.path = path;
        this.df = new TreeMap<>();

        if (proportions.isEmpty() || proportions.size() > 3) {
            throw new IllegalArgumentException("proportions must have 1-3 elements " +
                    "denoting the train:validation:test ratio");
        }
        this.proportions = proportions;
    }

    /** Reads the .csv file with data and converts it into a map
     * of the required form, keyed by {@code Headers}.
     *
     * @return The dataframe.
     * @throws RuntimeException If the path cannot be found or if the .csv file
     * is not of the correct format.
     */
    NavigableMap<Header, List<String>> csvToMap() {

        NavigableMap<Header, List<String>> newDf = new TreeMap<>();

        try (Reader in = new FileReader(path)) {

            // We set the headers to be Strings in the CSVRecord,
            // but we can convert these to enums later
            if (csvFormat == null) {
                csvFormat = CSVFormat.DEFAULT.builder()
                        .build();
            }

            Iterable<CSVRecord> records = csvFormat.parse(in);
            int dimensions = records.iterator().next().size() - 1;
            Header.setDimensions(dimensions);
            List<Header> headers = Header.getInitialHeaders();

            for (CSVRecord record : records) {
                for (int i = 0; i < dimensions + 1; i ++) {
                    Header header = headers.get(i);
                    newDf.computeIfAbsent(header, k -> new ArrayList<>())
                            .add(record.get(i));
                }
            }

        } catch (IOException e){
            throw new RuntimeException("Path %s is invalid.".formatted(path));
        }

        return newDf;
    }

    /** Main method for the class - splits the data into train:valid:test.
     *
     * @return A list containing the training, validation and testing dataframes
     *         or fewer, if fewer proportions have been passed.
     */
    public List<NavigableMap<Header, List<String>>> split() {
        // Read the .csv file
        df.putAll(csvToMap());

        int n = df.get(Header.Y).size();
        int propTotal = proportions.stream().mapToInt(Integer::intValue).sum();
        List<Integer> splits = new ArrayList<>(List.of(0));
        List<NavigableMap<Header, List<String>>> dfs = new ArrayList<>();

        for (int i = 0; i < proportions.size() - 1; i ++) {
            // Gets the length of this dataframe based on the proportions
            int lenNewDf = (int) (n * ((double) proportions.get(i) / propTotal));

            if (lenNewDf == 0) {
                lenNewDf = 1;
            }
            splits.add(splits.stream().mapToInt(Integer::intValue).sum() + lenNewDf);

            // The below delegates the task of getting the subset of df to another
            // method, getSubDf
            dfs.add(getSubDf(splits.get(i), splits.get(i + 1)));
        }
        int finalSplit = splits.get(splits.size() - 1);
        dfs.add(getSubDf(finalSplit, n));

        return dfs;
    }

    /** Returns a subset of the dataframe {@code df} from entry {@code fromIndex}
     * (inclusive) to entry {@code toIndex} (exclusive).
     *
     * @param fromIndex Starting index (inclusive).
     * @param toIndex Ending index (exclusive).
     * @return The subset of {@code df}.
     */
    NavigableMap<Header, List<String>> getSubDf(int fromIndex, int toIndex) {
        NavigableMap<Header, List<String>> subDf = new TreeMap<>();
        df.keySet()
                .forEach(header -> subDf.put(header,
                        df.get(header).subList(fromIndex, toIndex)));
        return subDf;
    }

    /** Getter method for {@code csvFormat}.
     *
     * @return The {@code csvFormat}. This is used for testing purposes.
     */
    CSVFormat getCsvFormat() {
        return csvFormat;
    }

    /** Setter method for {@code csvFormat}.
     *
     * @param csvFormat The new {@code csvFormat}. This can be used for mocking
     *                purposes.
     */
    void setCsvFormat(CSVFormat csvFormat) {
        this.csvFormat = csvFormat;
    }
}
