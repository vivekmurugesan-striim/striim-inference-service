package com.striim.inference.feast;

import com.fasterxml.jackson.databind.JsonNode;
import com.striim.inference.data.model.TransactionFeatureRecord;
import com.striim.inference.data.model.TransactionRecord;

import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Enhances a {@link TransactionRecord} into a {@link TransactionFeatureRecord}
 * carrying every feature the fraud-scoring model expects, by combining features
 * derived from the transaction itself with pre-computed features looked up from
 * Feast on the Customer, Merchant, Category, Customer&times;Merchant and
 * Customer&times;Merchant&times;Category entities.
 *
 * <p>The model consumes one named input per feature. The numeric inputs
 * correspond to {@link Feature}; the two categorical inputs are
 * {@code CATEGORY} (the transaction's merchant category, supplied as scoring
 * context) and {@code TopSpentCategory} (looked up from the customer feature
 * view). All Feast values are fetched in a single batched online request.
 */
public class FeastFeatureGenerator {

    // --- Feast entity join keys -------------------------------------------------
    private static final String CUSTOMER_ID = "customer_id";
    private static final String MERCHANT_ID = "merchant_id";
    private static final String CATEGORY = "category";

    // --- Feast feature views ----------------------------------------------------
    private static final String CUSTOMER_FV = "customer_features";
    private static final String MERCHANT_FV = "merchant_features";
    private static final String CATEGORY_FV = "merchant_category_features";
    private static final String CUST_MERCH_FV = "customer_merchant_features";
    private static final String CUST_MERCH_CAT_FV = "customer_merchant_category_features";

    // Contextual features looked up from Feast (not part of the numeric vector):
    //   - the customer's top-spent category feeds the TopSpentCategory model input,
    //   - the merchant's CATEGORY becomes the CATEGORY model input AND the join key
    //     for the category-based views,
    //   - the customer and merchant CITY derive the InCityTransaction flag.
    private static final String TOP_SPENT_CATEGORY_REF = CUSTOMER_FV + ":TopSpentCategory";
    private static final String CUSTOMER_CITY_REF = CUSTOMER_FV + ":CITY";
    private static final String MERCHANT_CATEGORY_REF = MERCHANT_FV + ":CATEGORY";
    private static final String MERCHANT_CITY_REF = MERCHANT_FV + ":CITY";

    /**
     * Ordered definition of the model's numeric features. Each entry carries the
     * ONNX input name the value is fed to, plus (for pre-computed features) the
     * Feast {@code <feature_view>:<feature_name>} reference. Transaction-derived
     * features carry a {@code null} reference and are computed locally.
     */
    public enum Feature {
        VALUE_USD("VALUEUSD", null),
        TRANSACTION_HOUR("TransactionHour", null),
        TRANSACTION_DAY_OF_WEEK("TransactionDayOfWeek", null),
        TRANSACTION_IS_WEEKEND("TransactionIsWeekend", null),
        TRANSACTION_MONTH("TransactionMonth", null),
        IN_CITY_TRANSACTION("InCityTransaction", null),

        ACCOUNT_AGE_DAYS("ACCOUNT_AGE_DAYS", CUSTOMER_FV + ":ACCOUNT_AGE_DAYS"),
        CUSTOMER_AVG_AMOUNT("CustomerAvgAmount", CUSTOMER_FV + ":CustomerAvgAmount"),
        HIGHEST_TRANSACTION_VALUE("HighestTransactionValue", CUSTOMER_FV + ":HighestTransactionValue"),
        CUSTOMER_FRAUD_RATE("CustomerFraudRate", CUSTOMER_FV + ":CustomerFraudRate"),
        CUSTOMER_DEVICE_COUNT("CustomerDeviceCount", CUSTOMER_FV + ":CustomerDeviceCount"),

        MERCHANT_AVG_AMOUNT("MerchantAvgAmount", MERCHANT_FV + ":MerchantAvgAmount"),
        HIGHEST_TRANSACTION_VALUE_FOR_MERCHANT("HighestTransactionValueForMerchant", MERCHANT_FV + ":HighestTransactionValueForMerchant"),
        MERCHANT_FRAUD_RATE("MerchantFraudRate", MERCHANT_FV + ":MerchantFraudRate"),

        CATEGORY_AVG_AMOUNT("CategoryAvgAmount", CATEGORY_FV + ":CategoryAvgAmount"),
        HIGHEST_TRANSACTION_VALUE_FOR_CATEGORY("HighestTransactionValueForCategory", CATEGORY_FV + ":HighestTransactionValueForCategory"),
        CATEGORY_FRAUD_RATE("CategoryFraudRate", CATEGORY_FV + ":CategoryFraudRate"),

        CUST_MERCH_AVG_AMOUNT("CustMerchAvgAmount", CUST_MERCH_FV + ":CustMerchAvgAmount"),
        HIGHEST_TRANSACTION_VALUE_FOR_MERCHANT_BY_CUSTOMER("HighestTransactionValueForMerchantByCustomer", CUST_MERCH_FV + ":HighestTransactionValueForMerchantByCustomer"),
        CUST_MERCH_FRAUD_RATE("CustMerchFraudRate", CUST_MERCH_FV + ":CustMerchFraudRate"),
        TIME_SINCE_LAST_TX("TimeSinceLastTx", CUST_MERCH_FV + ":TimeSinceLastTx"),

        CUST_MERCH_CAT_AVG_AMOUNT("CustMerchCatAvgAmount", CUST_MERCH_CAT_FV + ":CustMerchCatAvgAmount"),
        CUST_MERCH_CAT_FRAUD_RATE("CustMerchCatFraudRate", CUST_MERCH_CAT_FV + ":CustMerchCatFraudRate");

        private final String onnxInputName;
        private final String featureRef;

        Feature(String onnxInputName, String featureRef) {
            this.onnxInputName = onnxInputName;
            this.featureRef = featureRef;
        }

        /** Name of the ONNX model input this feature is fed to. */
        public String onnxInputName() {
            return onnxInputName;
        }

        /** Feast feature reference, or {@code null} for transaction-derived features. */
        public String featureRef() {
            return featureRef;
        }

        /** Feast feature name (without the view prefix), or {@code null}. */
        public String featureName() {
            return featureRef == null ? null : featureRef.substring(featureRef.indexOf(':') + 1);
        }

        /** Fully-qualified Feast name ("&lt;view&gt;__&lt;feature&gt;"), or {@code null}. */
        public String fullName() {
            return featureRef == null ? null : FeastFeatureGenerator.fullName(featureRef);
        }

        public boolean isFeastBacked() {
            return featureRef != null;
        }

        /** True if this feature's view is keyed by the (merchant-derived) category. */
        boolean isCategoryKeyed() {
            return featureRef != null
                    && (featureRef.startsWith(CATEGORY_FV + ":")
                        || featureRef.startsWith(CUST_MERCH_CAT_FV + ":"));
        }
    }

    /** Converts a "&lt;view&gt;:&lt;feature&gt;" reference to Feast's full name form. */
    private static String fullName(String featureRef) {
        return featureRef.replace(":", "__");
    }

    public static final int FEATURE_COUNT = Feature.values().length;

    private final FeastOnlineClient feastClient;
    private final ZoneId zoneId;

    public FeastFeatureGenerator(FeastOnlineClient feastClient) {
        this(feastClient, ZoneId.systemDefault());
    }

    public FeastFeatureGenerator(FeastOnlineClient feastClient, ZoneId zoneId) {
        this.feastClient = feastClient;
        this.zoneId = zoneId;
    }

    /**
     * Enhances a transaction with every model feature. Transaction-derived
     * features are computed locally; pre-computed features and scoring context
     * are fetched from Feast.
     *
     * <p>The merchant category and the in-city flag are <em>derived from Feast</em>,
     * not passed in:
     * <ul>
     *   <li>the {@code CATEGORY} input is the merchant's category
     *       ({@code merchant_features:CATEGORY});</li>
     *   <li>{@code InCityTransaction} is {@code 1} when the customer's city equals
     *       the merchant's city, else {@code 0}.</li>
     * </ul>
     *
     * <p>Because the merchant category is also the join key for the category-based
     * views, the lookup is done in two phases: first the customer / merchant /
     * customer&times;merchant features (which yield the category and both cities),
     * then the category-keyed views using the derived category.
     *
     * @param record the transaction being scored
     * @return the enhanced feature record ready for inference
     */
    public TransactionFeatureRecord generate(TransactionRecord record) {
        LocalDateTime txTime = LocalDateTime.ofInstant(record.getTimestamp().toInstant(), zoneId);

        // --- Phase 1: everything keyed by customer_id / merchant_id -------------
        List<String> phase1Refs = new ArrayList<>();
        for (Feature f : Feature.values()) {
            if (f.isFeastBacked() && !f.isCategoryKeyed()) {
                phase1Refs.add(f.featureRef());
            }
        }
        phase1Refs.add(TOP_SPENT_CATEGORY_REF);
        phase1Refs.add(CUSTOMER_CITY_REF);
        phase1Refs.add(MERCHANT_CATEGORY_REF);
        phase1Refs.add(MERCHANT_CITY_REF);

        Map<String, Object> phase1Entities = new LinkedHashMap<>();
        phase1Entities.put(CUSTOMER_ID, record.getCustId());
        phase1Entities.put(MERCHANT_ID, record.getMerchantId());

        Map<String, JsonNode> feast = new HashMap<>(
                feastClient.getOnlineFeatures(phase1Refs, phase1Entities));

        // Derive category + in-city flag from the phase-1 result.
        String category = readFeastText(feast, fullName(MERCHANT_CATEGORY_REF));
        String customerCity = readFeastText(feast, fullName(CUSTOMER_CITY_REF));
        String merchantCity = readFeastText(feast, fullName(MERCHANT_CITY_REF));
        boolean inCityTransaction = !customerCity.isEmpty() && customerCity.equals(merchantCity);

        // --- Phase 2: category-keyed views, now that the category is known ------
        List<String> phase2Refs = new ArrayList<>();
        for (Feature f : Feature.values()) {
            if (f.isCategoryKeyed()) {
                phase2Refs.add(f.featureRef());
            }
        }
        if (!category.isEmpty() && !phase2Refs.isEmpty()) {
            Map<String, Object> phase2Entities = new LinkedHashMap<>();
            phase2Entities.put(CUSTOMER_ID, record.getCustId());
            phase2Entities.put(CATEGORY, category);
            feast.putAll(feastClient.getOnlineFeatures(phase2Refs, phase2Entities));
        }

        Map<String, Float> numeric = new LinkedHashMap<>();
        for (Feature f : Feature.values()) {
            float value = switch (f) {
                case VALUE_USD -> record.getValue();
                case TRANSACTION_HOUR -> txTime.getHour();
                case TRANSACTION_DAY_OF_WEEK -> txTime.getDayOfWeek().getValue(); // Mon=1 .. Sun=7
                case TRANSACTION_IS_WEEKEND -> isWeekend(txTime.getDayOfWeek()) ? 1f : 0f;
                case TRANSACTION_MONTH -> txTime.getMonthValue();
                case IN_CITY_TRANSACTION -> inCityTransaction ? 1f : 0f;
                default -> readFeastFloat(feast, f.fullName());
            };
            numeric.put(f.onnxInputName(), value);
        }

        Map<String, String> categorical = new LinkedHashMap<>();
        categorical.put(TransactionFeatureRecord.INPUT_CATEGORY, category);
        categorical.put(TransactionFeatureRecord.INPUT_TOP_SPENT_CATEGORY,
                readFeastText(feast, fullName(TOP_SPENT_CATEGORY_REF)));

        return new TransactionFeatureRecord(record.getId(), numeric, categorical);
    }

    /**
     * Builds the numeric feature vector in {@link Feature} order. Retained for
     * debugging / callers that want the raw ordered array rather than a record.
     */
    public float[] generateFeatures(TransactionRecord record) {
        Map<String, Float> numeric = generate(record).getNumericFeatures();
        float[] out = new float[FEATURE_COUNT];
        for (Feature f : Feature.values()) {
            out[f.ordinal()] = numeric.get(f.onnxInputName());
        }
        return out;
    }

    private static boolean isWeekend(DayOfWeek day) {
        return day == DayOfWeek.SATURDAY || day == DayOfWeek.SUNDAY;
    }

    /**
     * Reads a Feast-backed numeric feature, defaulting missing values to
     * {@code 0.0f} (matching how the model treats an absent cold-start feature).
     */
    private static float readFeastFloat(Map<String, JsonNode> feast, String featureName) {
        JsonNode value = feast.get(featureName);
        if (value == null || value.isNull()) {
            return 0.0f;
        }
        return (float) value.asDouble();
    }

    /** Reads a Feast-backed string feature, defaulting missing values to {@code ""}. */
    private static String readFeastText(Map<String, JsonNode> feast, String featureName) {
        JsonNode value = feast.get(featureName);
        if (value == null || value.isNull()) {
            return "";
        }
        return value.asText();
    }
}
