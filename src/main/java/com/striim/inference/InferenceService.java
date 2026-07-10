package com.striim.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.striim.inference.config.InferenceConfig;
import com.striim.inference.data.model.PredictionResult;
import com.striim.inference.data.model.TransactionFeatureRecord;
import com.striim.inference.data.model.TransactionRecord;
import com.striim.inference.feast.FeastFeatureGenerator;
import com.striim.inference.feast.FeastOnlineClient;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Fraud-scoring inference service.
 *
 * <p>Loads the ONNX model (converted from a Spark ML pipeline) and scores
 * transactions. Given a raw {@link TransactionRecord}, it enhances the record
 * into a {@link TransactionFeatureRecord} via {@link FeastFeatureGenerator}
 * (looking features up from Feast), binds each feature to the matching ONNX
 * model input, runs inference and returns the {@link PredictionResult}.
 *
 * <p>The ONNX model exposes one named input per feature: 23 numeric (float)
 * inputs plus two categorical (string) inputs, {@code CATEGORY} and
 * {@code TopSpentCategory}. Outputs are {@code prediction} (class) and
 * {@code probability} (per-class probabilities).
 *
 * <p>Not thread-safe for concurrent {@link #infer} calls sharing one session is
 * fine (ONNX Runtime sessions are thread-safe), but this instance owns the
 * session lifecycle — call {@link #close()} when done.
 */
public class InferenceService implements AutoCloseable {

    private static final String OUTPUT_PREDICTION = "prediction";
    private static final String OUTPUT_PROBABILITY = "probability";

    private static final ObjectMapper JSON = new ObjectMapper();

    private final OrtEnvironment env;
    private final OrtSession session;
    private final FeastFeatureGenerator featureGenerator;

    /**
     * Builds a service from configuration: loads the ONNX model and wires a
     * Feast-backed feature generator pointing at the configured feature server.
     */
    public InferenceService(InferenceConfig config) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(config.getOnnxModelPath(), new OrtSession.SessionOptions());
        this.featureGenerator =
                new FeastFeatureGenerator(new FeastOnlineClient(config.getFeastBaseUrl()));
    }

    /**
     * Builds a service with an explicit feature generator. Useful for tests
     * (e.g. a generator backed by a mock Feast client) and dependency injection.
     */
    public InferenceService(OrtEnvironment env,
                            OrtSession session,
                            FeastFeatureGenerator featureGenerator) {
        this.env = env;
        this.session = session;
        this.featureGenerator = featureGenerator;
    }

    /**
     * One-shot static scoring: loads configuration from the given file, creates
     * an {@link InferenceService}, scores the transaction and returns the result
     * as a JSON string. The service (and its ONNX session) is created and closed
     * within the call.
     *
     * @param absPathOfConfigFile absolute path to an {@code inference.properties} file
     * @param inputs              raw transaction fields, as for {@link #predict(Object...)}
     * @return a JSON representation of the {@link PredictionResult}, e.g.
     *         {@code {"prediction":0,"probabilities":[0.61,0.39],"fraud":false,"fraudProbability":0.39}}
     */
    public static String predictJson(String absPathOfConfigFile,
    Object... inputs)
            throws OrtException {
        InferenceConfig config = InferenceConfig.load(Path.of(absPathOfConfigFile));
        try (InferenceService service = new InferenceService(config)) {
            PredictionResult result = service.predict(inputs);
            try {
                return JSON.writeValueAsString(result);
            } catch (JsonProcessingException e) {
                throw new IllegalStateException("Failed to serialize PredictionResult to JSON", e);
            }
        }
    }

    /**
     * End-to-end scoring from raw transaction fields.
     *
     * <p>The arguments are parsed into a {@link TransactionRecord} in the order
     * expected by {@link TransactionRecord#parse(Object...)}:
     * {@code id, status, timestamp, value, custId, deviceId, merchantId}. The
     * merchant category and the in-city flag are derived from Feast, so they are
     * not passed here.
     *
     * @param inputs raw transaction fields
     */
    public PredictionResult predict(Object... inputs) throws OrtException {
        return predict(TransactionRecord.parse(inputs));
    }

    /**
     * End-to-end scoring: enhance the transaction with Feast features (which also
     * derive the merchant category and the in-city flag), then run inference.
     *
     * @param record the raw transaction
     */
    public PredictionResult predict(TransactionRecord record) throws OrtException {
        if (featureGenerator == null) {
            throw new IllegalStateException("No FeastFeatureGenerator configured; use infer(...) "
                    + "with a pre-built TransactionFeatureRecord.");
        }
        return infer(featureGenerator.generate(record));
    }

    /**
     * Runs inference on an already-enhanced feature record by binding every model
     * input to its value (string tensor for categorical inputs, float tensor
     * otherwise).
     */
    public PredictionResult infer(TransactionFeatureRecord features) throws OrtException {
        debugPrintFeatures(features);
        Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
        List<OnnxTensor> created = new ArrayList<>();
        try {
            for (String inputName : session.getInputNames()) {
                OnnxTensor tensor = buildTensor(inputName, features);
                inputs.put(inputName, tensor);
                created.add(tensor);
            }

            try (Result output = session.run(inputs)) {
                long[] prediction = (long[]) tensorValue(output, OUTPUT_PREDICTION);
                float[][] probability = (float[][]) tensorValue(output, OUTPUT_PROBABILITY);
                return new PredictionResult(prediction[0], probability[0]);
            }
        } finally {
            for (OnnxTensor t : created) {
                t.close();
            }
        }
    }

    /**
     * Prints the fully-enhanced feature record — every model input and its value,
     * in the model's input order — right before inference is run.
     */
    private void debugPrintFeatures(TransactionFeatureRecord features) throws OrtException {
        StringBuilder sb = new StringBuilder();
        sb.append("\n--- MODEL INPUT (transactionId=")
                .append(features.getTransactionId()).append(") ---\n");
        int i = 1;
        for (String inputName : session.getInputNames()) {
            Object value = features.isCategorical(inputName)
                    ? features.getCategoricalFeatures().get(inputName)
                    : features.getNumericFeatures().get(inputName);
            String type = features.isCategorical(inputName) ? "string" : "float";
            sb.append(String.format("  %2d. %-46s = %-16s (%s)%n",
                    i++, inputName, value, type));
        }
        sb.append("-------------------------------------------");
        System.out.println(sb);
    }

    private OnnxTensor buildTensor(String inputName, TransactionFeatureRecord features)
            throws OrtException {
        if (features.isCategorical(inputName)) {
            String value = features.getCategoricalFeatures().get(inputName);
            // Categorical model inputs have shape [1, 1].
            return OnnxTensor.createTensor(env, new String[][]{{value == null ? "" : value}});
        }
        Float value = features.getNumericFeatures().get(inputName);
        if (value == null) {
            throw new IllegalArgumentException(
                    "Missing value for model input '" + inputName + "' in " + features);
        }
        return OnnxTensor.createTensor(env,
                FloatBuffer.wrap(new float[]{value}), new long[]{1, 1});
    }

    private static Object tensorValue(Result output, String name) throws OrtException {
        return ((OnnxTensor) output.get(name)
                .orElseThrow(() -> new IllegalStateException("Model has no output '" + name + "'")))
                .getValue();
    }

    @Override
    public void close() throws OrtException {
        // Close the session we own. The OrtEnvironment is a shared process-wide
        // singleton, so it is intentionally not closed here.
        if (session != null) {
            session.close();
        }
    }

    /**
     * Demo entrypoint: loads config, scores a sample transaction end-to-end
     * against the configured Feast feature server. Requires the Feast container
     * (see {@code feast/docker-compose.yml}) to be running.
     */
    public static void main(String[] args) throws Exception {
        InferenceConfig config = InferenceConfig.load();
        System.out.println("Starting inference service with " + config);

        try (InferenceService service = new InferenceService(config)) {
            // Raw transaction fields: id, status, timestamp, value, custId, deviceId, merchantId.
            // Category and in-city flag are derived from Feast, not passed here.
            PredictionResult result = service.predict(
                    1001L, "PENDING", "2026-07-07 13:00:00.000",
                    80.29706738567387, /* custId */ 1L, /* deviceId */ 1L, /* merchantId */ 1L);

            System.out.println("\n--- INFERENCE RESULT ---");
            System.out.println(result);
            System.out.println("------------------------");
        }
    }
}
