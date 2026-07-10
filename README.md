# InferenceService

Fraud-scoring inference service. Given a raw transaction, it enriches the record
with pre-computed features looked up from a [Feast](https://feast.dev) online
feature store, runs the enriched feature set through an ONNX fraud model, and
returns the prediction and per-class probabilities.

```
TransactionRecord ──▶ FeastFeatureGenerator ──▶ TransactionFeatureRecord ──▶ InferenceService ──▶ PredictionResult
                          (Feast lookup)            (all model features)         (model.onnx)      (prediction + probs)
```

## Contents

| Path | Purpose |
|------|---------|
| [`src/main/java/com/striim/inference/InferenceService.java`](src/main/java/com/striim/inference/InferenceService.java) | Loads the ONNX model, enriches + scores transactions |
| [`src/main/java/com/striim/inference/feast/FeastFeatureGenerator.java`](src/main/java/com/striim/inference/feast/FeastFeatureGenerator.java) | Builds all model features (transaction-derived + Feast) |
| [`src/main/java/com/striim/inference/feast/FeastOnlineClient.java`](src/main/java/com/striim/inference/feast/FeastOnlineClient.java) | HTTP client for the Feast online serving API |
| [`src/main/java/com/striim/inference/config/InferenceConfig.java`](src/main/java/com/striim/inference/config/InferenceConfig.java) | Configuration (ONNX path, Feast URL) |
| [`src/main/resources/inference.properties`](src/main/resources/inference.properties) | Default configuration values |
| [`model.onnx`](model.onnx) | The fraud-scoring model (Spark ML pipeline → ONNX) |
| [`feast/`](feast/) | Feast feature store: CSV loader + Docker (see [feast/README.md](feast/README.md)) |

## Prerequisites

- **JDK 17+**
- **Maven 3.8+**
- **Docker** (to run the Feast feature store)
- `model.onnx` present in the project root (already included)

## The model

`model.onnx` exposes **one named input per feature** — 23 numeric (float) inputs
plus two categorical (string) inputs, `CATEGORY` and `TopSpentCategory` — and
produces `prediction` (`0`=non-fraud, `1`=fraud) and `probability`
(`[P(non-fraud), P(fraud)]`). The 23 numeric inputs are:

1. `VALUEUSD` · 2. `TransactionHour` · 3. `TransactionDayOfWeek` ·
4. `TransactionIsWeekend` · 5. `TransactionMonth` · 6. `InCityTransaction` ·
7. `ACCOUNT_AGE_DAYS` · 8. `CustomerAvgAmount` · 9. `HighestTransactionValue` ·
10. `CustomerFraudRate` · 11. `CustomerDeviceCount` · 12. `MerchantAvgAmount` ·
13. `HighestTransactionValueForMerchant` · 14. `MerchantFraudRate` ·
15. `CategoryAvgAmount` · 16. `HighestTransactionValueForCategory` ·
17. `CategoryFraudRate` · 18. `CustMerchAvgAmount` ·
19. `HighestTransactionValueForMerchantByCustomer` · 20. `CustMerchFraudRate` ·
21. `TimeSinceLastTx` · 22. `CustMerchCatAvgAmount` · 23. `CustMerchCatFraudRate`

Features 1–6 are derived from the transaction; the rest (plus `TopSpentCategory`)
are looked up from Feast.

## Configuration

Configured via [`inference.properties`](src/main/resources/inference.properties),
with each value overridable by an environment variable (env wins):

| Property | Env var | Default | Meaning |
|----------|---------|---------|---------|
| `onnx.model.path` | `ONNX_MODEL_PATH` | `./model.onnx` | Path to the ONNX model binary |
| `feast.base.url` | `FEAST_BASE_URL` | `http://localhost:6566` | Feast online server base URL |

## Running

### 1. Start the Feast feature store

The service needs the Feast online server running with feature data loaded. From
the project root (point `FEATURES_HOME` at the directory holding the
`CustomerFeatures/`, `MerchantFeatures/`, … CSV sub-directories):

```bash
FEATURES_HOME=/abs/path/to/Features docker compose -f feast/docker-compose.yml up --build
```

This loads the CSVs, materializes them, and serves on `http://localhost:6566`.
See [feast/README.md](feast/README.md) for details. Verify it is up:

```bash
curl -s http://localhost:6566/get-online-features \
  -H 'Content-Type: application/json' \
  -d '{"features":["customer_features:CustomerAvgAmount"],"entities":{"customer_id":[1]}}'
```

### 2. Run the inference service

The `main` method scores a sample transaction end-to-end against Feast:

```bash
mvn compile exec:java
```

Override configuration at runtime as needed:

```bash
FEAST_BASE_URL=http://localhost:6566 ONNX_MODEL_PATH=./model.onnx mvn compile exec:java
```

Expected output:

```
Starting inference service with InferenceConfig{onnxModelPath='./model.onnx', feastBaseUrl='http://localhost:6566'}
--- INFERENCE RESULT ---
PredictionResult{prediction=0 (NON-FRAUD), probabilities=[0.50, 0.50]}
------------------------
```

### As a self-contained jar

`mvn package` builds an executable uber-jar with all dependencies (including the
ONNX Runtime native libraries) bundled:

```bash
mvn package -DskipTests
java -jar target/InferenceService-1.0-SNAPSHOT-shaded.jar
```

Configuration still comes from `inference.properties` / the `ONNX_MODEL_PATH` and
`FEAST_BASE_URL` environment variables.

## Using the service in code

```java
InferenceConfig config = InferenceConfig.load();
try (InferenceService service = new InferenceService(config)) {
    // Raw transaction fields: id, status, timestamp, value, custId, deviceId, merchantId.
    // The merchant category and the in-city flag are derived from Feast, not passed in.
    PredictionResult result = service.predict(
            1001L, "PENDING", "2026-07-07 13:00:00.000",
            80.29706738567387, /* custId */ 1L, /* deviceId */ 1L, /* merchantId */ 1L);

    System.out.println(result.isFraud());              // true / false
    System.out.println(result.getFraudProbability());  // P(fraud)
}
```

`predict(Object...)` parses the fields into a `TransactionRecord`; a typed
`predict(TransactionRecord)` overload is also available. During feature
enhancement the service looks up the merchant's category
(`merchant_features:CATEGORY`) for the `CATEGORY` input and the category-based
views, and sets `InCityTransaction = 1` when the customer's city equals the
merchant's city (`customer_features:CITY` vs `merchant_features:CITY`).

To score a pre-built feature set without a Feast lookup, call
`service.infer(TransactionFeatureRecord)` directly.

## Testing

```bash
mvn test
```

[`InferenceServiceTest`](src/test/java/com/striim/inference/InferenceServiceTest.java)
runs against the real `model.onnx` and does **not** require a live Feast server:

- `infersFromMockFeatureRecord` — scores a hand-built feature record.
- `predictEnhancesViaFeastThenInfers` — full path with a mock Feast client, so it
  runs fully offline (e.g. in CI).

## Deploying alongside Feast (same Docker network)

When the service itself runs in a container on the same network as the Feast
container, point it at the service name instead of `localhost`:

```bash
FEAST_BASE_URL=http://feast:6566
```
