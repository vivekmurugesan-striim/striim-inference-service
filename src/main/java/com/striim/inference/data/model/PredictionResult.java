package com.striim.inference.data.model;

import java.util.Arrays;

/**
 * Outcome of a single fraud-scoring inference: the model's class prediction and
 * the per-class probabilities.
 */
public class PredictionResult {

    private final long prediction;         // 0 = non-fraud, 1 = fraud
    private final float[] probabilities;   // [P(non-fraud), P(fraud)]

    public PredictionResult(long prediction, float[] probabilities) {
        this.prediction = prediction;
        this.probabilities = probabilities.clone();
    }

    /** Predicted class: {@code 0} = non-fraud, {@code 1} = fraud. */
    public long getPrediction() {
        return prediction;
    }

    /** Per-class probabilities: index 0 = non-fraud, index 1 = fraud. */
    public float[] getProbabilities() {
        return probabilities.clone();
    }

    public boolean isFraud() {
        return prediction == 1L;
    }

    /** Convenience accessor for the fraud probability (class 1). */
    public float getFraudProbability() {
        return probabilities.length > 1 ? probabilities[1] : Float.NaN;
    }

    @Override
    public String toString() {
        return "PredictionResult{prediction=" + prediction
                + " (" + (isFraud() ? "FRAUD" : "NON-FRAUD") + ")"
                + ", probabilities=" + Arrays.toString(probabilities) + '}';
    }
}
