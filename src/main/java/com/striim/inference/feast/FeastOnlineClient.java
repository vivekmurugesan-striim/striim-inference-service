package com.striim.inference.feast;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Thin client over the Feast online feature server HTTP endpoint
 * ({@code POST /get-online-features}).
 *
 * <p>A single request may reference features from several feature views; Feast
 * resolves the join keys each view needs from the supplied entity map, so all
 * entities for a lookup can be sent together in one round trip. Feature
 * references use the Feast convention {@code <feature_view>:<feature_name>}.
 */
public class FeastOnlineClient {

    private final String endpoint;
    private final HttpClient httpClient;
    private final ObjectMapper mapper = new ObjectMapper();

    public FeastOnlineClient(String baseUrl) {
        this(baseUrl, HttpClient.newBuilder()
                // Feast's server (uvicorn) speaks HTTP/1.1 only. The JDK client
                // defaults to HTTP/2 and attempts an h2c cleartext upgrade, which
                // uvicorn rejects with "400 Invalid HTTP request received".
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(5))
                .build());
    }

    public FeastOnlineClient(String baseUrl, HttpClient httpClient) {
        String trimmed = baseUrl.endsWith("/")
                ? baseUrl.substring(0, baseUrl.length() - 1)
                : baseUrl;
        this.endpoint = trimmed + "/get-online-features";
        this.httpClient = httpClient;
    }

    /**
     * Fetches the given feature references for a single entity row.
     *
     * @param featureRefs feature references, e.g. {@code "customer_features:customer_avg_amount"}
     * @param entities    join-key -> value for the single row being scored
     * @return map of feature name (without the view prefix) -> value, missing
     * features omitted from the map
     */
    public Map<String, JsonNode> getOnlineFeatures(List<String> featureRefs,
                                                   Map<String, Object> entities) {
        String body = buildRequestBody(featureRefs, entities);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .timeout(Duration.ofSeconds(10))
                .version(HttpClient.Version.HTTP_1_1)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response;
        try {
            response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (Exception e) {
            throw new FeastLookupException("Failed to call Feast at " + endpoint, e);
        }

        if (response.statusCode() / 100 != 2) {
            throw new FeastLookupException("Feast returned HTTP " + response.statusCode()
                    + ": " + response.body());
        }
        return parseResponse(response.body());
    }

    private String buildRequestBody(List<String> featureRefs, Map<String, Object> entities) {
        ObjectNode root = mapper.createObjectNode();

        // Return fully-qualified feature names ("<view>__<feature>") so features
        // that share a short name across views (e.g. CITY, CATEGORY) don't collide.
        root.put("full_feature_names", true);

        ArrayNode features = root.putArray("features");
        for (String ref : featureRefs) {
            features.add(ref);
        }

        ObjectNode entityNode = root.putObject("entities");
        for (Map.Entry<String, Object> e : entities.entrySet()) {
            // The online API expects each entity value as a list (one entry per row).
            ArrayNode values = entityNode.putArray(e.getKey());
            Object v = e.getValue();
            if (v instanceof Number n) {
                values.add(n.longValue());
            } else {
                values.add(String.valueOf(v));
            }
        }

        try {
            return mapper.writeValueAsString(root);
        } catch (Exception e) {
            throw new FeastLookupException("Failed to serialize Feast request", e);
        }
    }

    /**
     * Feast response shape:
     * <pre>
     * {
     *   "metadata": { "feature_names": ["customer_id", "customer_avg_amount", ...] },
     *   "results":  [ { "values": [123], "statuses": ["PRESENT"], ... }, ... ]
     * }
     * </pre>
     * The i-th entry in {@code results} aligns with the i-th name in
     * {@code feature_names}; {@code values[0]} is this row's value.
     */
    private Map<String, JsonNode> parseResponse(String body) {
        Map<String, JsonNode> out = new HashMap<>();
        JsonNode root;
        try {
            root = mapper.readTree(body);
        } catch (Exception e) {
            throw new FeastLookupException("Failed to parse Feast response: " + body, e);
        }

        JsonNode names = root.path("metadata").path("feature_names");
        JsonNode results = root.path("results");
        if (!names.isArray() || !results.isArray()) {
            throw new FeastLookupException("Unexpected Feast response shape: " + body);
        }

        for (int i = 0; i < names.size() && i < results.size(); i++) {
            String name = names.get(i).asText();
            JsonNode values = results.get(i).path("values");
            JsonNode statuses = results.get(i).path("statuses");
            if (!values.isArray() || values.isEmpty()) {
                continue;
            }
            String status = statuses.isArray() && !statuses.isEmpty()
                    ? statuses.get(0).asText() : "PRESENT";
            JsonNode value = values.get(0);
            if (value.isNull() || "NOT_FOUND".equals(status)) {
                continue;
            }
            out.put(name, value);
        }
        return out;
    }

    /** Unchecked failure raised for any transport or protocol error talking to Feast. */
    public static class FeastLookupException extends RuntimeException {
        public FeastLookupException(String message) {
            super(message);
        }

        public FeastLookupException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
