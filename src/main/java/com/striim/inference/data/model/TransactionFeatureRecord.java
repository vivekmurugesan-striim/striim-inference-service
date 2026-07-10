package com.striim.inference.data.model;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A {@link TransactionRecord} enhanced with every feature the fraud-scoring
 * ONNX model requires. Features are keyed by their ONNX model input name so the
 * inference service can bind each value to the matching model input regardless
 * of ordering.
 *
 * <ul>
 *   <li>{@link #getNumericFeatures()} — the 23 numeric model inputs
 *       (e.g. {@code VALUEUSD}, {@code CustomerAvgAmount}).</li>
 *   <li>{@link #getCategoricalFeatures()} — the string model inputs
 *       {@code CATEGORY} and {@code TopSpentCategory}.</li>
 * </ul>
 */
public class TransactionFeatureRecord {

    /** ONNX input name for the transaction's merchant category. */
    public static final String INPUT_CATEGORY = "CATEGORY";
    /** ONNX input name for the customer's top-spent category. */
    public static final String INPUT_TOP_SPENT_CATEGORY = "TopSpentCategory";

    private final long transactionId;
    private final Map<String, Float> numericFeatures;
    private final Map<String, String> categoricalFeatures;

    public TransactionFeatureRecord(long transactionId,
                                    Map<String, Float> numericFeatures,
                                    Map<String, String> categoricalFeatures) {
        this.transactionId = transactionId;
        this.numericFeatures = new LinkedHashMap<>(numericFeatures);
        this.categoricalFeatures = new LinkedHashMap<>(categoricalFeatures);
    }

    public long getTransactionId() {
        return transactionId;
    }

    /** Numeric model inputs, keyed by ONNX input name. Unmodifiable. */
    public Map<String, Float> getNumericFeatures() {
        return Collections.unmodifiableMap(numericFeatures);
    }

    /** Categorical (string) model inputs, keyed by ONNX input name. Unmodifiable. */
    public Map<String, String> getCategoricalFeatures() {
        return Collections.unmodifiableMap(categoricalFeatures);
    }

    /** @return true if the given ONNX input name is a categorical (string) input. */
    public boolean isCategorical(String onnxInputName) {
        return categoricalFeatures.containsKey(onnxInputName);
    }

    @Override
    public String toString() {
        return "TransactionFeatureRecord{transactionId=" + transactionId
                + ", numericFeatures=" + numericFeatures
                + ", categoricalFeatures=" + categoricalFeatures + '}';
    }
}
