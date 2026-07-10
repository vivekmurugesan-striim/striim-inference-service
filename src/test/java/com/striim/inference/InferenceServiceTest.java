package com.striim.inference;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.striim.inference.config.InferenceConfig;
import com.striim.inference.data.model.PredictionResult;
import com.striim.inference.data.model.TransactionFeatureRecord;
import com.striim.inference.feast.FeastFeatureGenerator;
import com.striim.inference.feast.FeastOnlineClient;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises {@link InferenceService} against the real {@code ./model.onnx}
 * binary using mock feature data (no live Feast server required).
 */
class InferenceServiceTest {

    private static InferenceConfig config;

    @BeforeAll
    static void setUp() {
        config = InferenceConfig.load();
        assumeTrue(Files.exists(Paths.get(config.getOnnxModelPath())),
                "model.onnx not found at " + config.getOnnxModelPath());
    }

    /**
     * Feeds a pre-built {@link TransactionFeatureRecord} straight into the model
     * and validates the shape of the prediction/probabilities.
     */
    @Test
    void infersFromMockFeatureRecord() throws Exception {
        try (InferenceService service = new InferenceService(config)) {
            PredictionResult result = service.infer(sampleFeatureRecord());

            assertNotNull(result);
            assertTrue(result.getPrediction() == 0L || result.getPrediction() == 1L,
                    "prediction must be 0 or 1, was " + result.getPrediction());

            float[] probs = result.getProbabilities();
            assertEquals(2, probs.length, "expected [P(non-fraud), P(fraud)]");
            for (float p : probs) {
                assertTrue(p >= 0f && p <= 1f, "probability out of range: " + p);
            }
            assertEquals(1.0f, probs[0] + probs[1], 1e-3f, "probabilities should sum to ~1");
            assertEquals(probs[1], result.getFraudProbability(), 0f);

            System.out.println("infersFromMockFeatureRecord -> " + result);
        }
    }

    /**
     * Exercises the full path: a raw {@link TransactionRecord} enhanced via
     * {@link FeastFeatureGenerator} (backed by a mock Feast client) and scored.
     */
    @Test
    void predictEnhancesViaFeastThenInfers() throws Exception {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession(config.getOnnxModelPath(),
                new OrtSession.SessionOptions());
        FeastFeatureGenerator generator = new FeastFeatureGenerator(new MockFeastClient());

        try (InferenceService service = new InferenceService(env, session, generator)) {
            // Uses the Object... API; category + in-city flag are derived from the
            // (mock) Feast values, not passed in.
            PredictionResult result = service.predict(
                    2002L, "PENDING", "2026-07-07 13:00:00.000",
                    80.29706738567387, /* custId */ 1L, /* deviceId */ 1L, /* merchantId */ 541L);

            assertNotNull(result);
            assertTrue(result.getPrediction() == 0L || result.getPrediction() == 1L);
            assertEquals(2, result.getProbabilities().length);
            System.out.println("predictEnhancesViaFeastThenInfers -> " + result);
        }
    }

    /** The 23 numeric inputs + 2 categorical inputs from the sample transaction. */
    private static TransactionFeatureRecord sampleFeatureRecord() {
        Map<String, Float> numeric = new LinkedHashMap<>();
        numeric.put("VALUEUSD", 80.29706738567387f);
        numeric.put("TransactionHour", 13f);
        numeric.put("TransactionDayOfWeek", 6f);
        numeric.put("TransactionIsWeekend", 1f);
        numeric.put("TransactionMonth", 1f);
        numeric.put("InCityTransaction", 0f);
        numeric.put("ACCOUNT_AGE_DAYS", 1711f);
        numeric.put("CustomerAvgAmount", 193.38621093749907f);
        numeric.put("HighestTransactionValue", 2310.424700314853f);
        numeric.put("CustomerFraudRate", 0.0f);
        numeric.put("CustomerDeviceCount", 1f);
        numeric.put("MerchantAvgAmount", 75.6564298703668f);
        numeric.put("HighestTransactionValueForMerchant", 193.5838482989632f);
        numeric.put("MerchantFraudRate", 0.0f);
        numeric.put("CategoryAvgAmount", 75.53530101885512f);
        numeric.put("HighestTransactionValueForCategory", 235.5710181209811f);
        numeric.put("CategoryFraudRate", 0.0f);
        numeric.put("CustMerchAvgAmount", 77.55696467939407f);
        numeric.put("HighestTransactionValueForMerchantByCustomer", 124.6042441253706f);
        numeric.put("CustMerchFraudRate", 0.0f);
        numeric.put("TimeSinceLastTx", 6357528.375f);
        numeric.put("CustMerchCatAvgAmount", 78.05308429412722f);
        numeric.put("CustMerchCatFraudRate", 0.0f);

        Map<String, String> categorical = new LinkedHashMap<>();
        categorical.put(TransactionFeatureRecord.INPUT_CATEGORY, "GROCERIES");
        categorical.put(TransactionFeatureRecord.INPUT_TOP_SPENT_CATEGORY, "INSURANCE");

        return new TransactionFeatureRecord(1001L, numeric, categorical);
    }

    /**
     * Stand-in for {@link FeastOnlineClient} that returns canned online feature
     * values instead of calling a live server. Keys use Feast's fully-qualified
     * "&lt;view&gt;__&lt;feature&gt;" form (full_feature_names=true). Customer and
     * merchant cities are equal so the derived InCityTransaction flag is true.
     */
    private static final class MockFeastClient extends FeastOnlineClient {
        private static final JsonNodeFactory NODES = JsonNodeFactory.instance;

        MockFeastClient() {
            super("http://mock-feast");
        }

        @Override
        public Map<String, JsonNode> getOnlineFeatures(List<String> featureRefs,
                                                       Map<String, Object> entities) {
            Map<String, JsonNode> f = new LinkedHashMap<>();
            // customer_features
            f.put("customer_features__ACCOUNT_AGE_DAYS", NODES.numberNode(1711.0));
            f.put("customer_features__CustomerAvgAmount", NODES.numberNode(193.38621093749907));
            f.put("customer_features__HighestTransactionValue", NODES.numberNode(2310.424700314853));
            f.put("customer_features__CustomerFraudRate", NODES.numberNode(0.0));
            f.put("customer_features__CustomerDeviceCount", NODES.numberNode(1.0));
            f.put("customer_features__TopSpentCategory", NODES.textNode("INSURANCE"));
            f.put("customer_features__CITY", NODES.textNode("LAS_VEGAS"));
            // merchant_features (CATEGORY -> CATEGORY input + category join key)
            f.put("merchant_features__MerchantAvgAmount", NODES.numberNode(75.6564298703668));
            f.put("merchant_features__HighestTransactionValueForMerchant", NODES.numberNode(193.5838482989632));
            f.put("merchant_features__MerchantFraudRate", NODES.numberNode(0.0));
            f.put("merchant_features__CATEGORY", NODES.textNode("GROCERIES"));
            f.put("merchant_features__CITY", NODES.textNode("LAS_VEGAS"));
            // merchant_category_features
            f.put("merchant_category_features__CategoryAvgAmount", NODES.numberNode(75.53530101885512));
            f.put("merchant_category_features__HighestTransactionValueForCategory", NODES.numberNode(235.5710181209811));
            f.put("merchant_category_features__CategoryFraudRate", NODES.numberNode(0.0));
            // customer_merchant_features
            f.put("customer_merchant_features__CustMerchAvgAmount", NODES.numberNode(77.55696467939407));
            f.put("customer_merchant_features__HighestTransactionValueForMerchantByCustomer", NODES.numberNode(124.6042441253706));
            f.put("customer_merchant_features__CustMerchFraudRate", NODES.numberNode(0.0));
            f.put("customer_merchant_features__TimeSinceLastTx", NODES.numberNode(6357528.375));
            // customer_merchant_category_features
            f.put("customer_merchant_category_features__CustMerchCatAvgAmount", NODES.numberNode(78.05308429412722));
            f.put("customer_merchant_category_features__CustMerchCatFraudRate", NODES.numberNode(0.0));
            return f;
        }
    }
}
