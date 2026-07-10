package com.striim.inference.config;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/**
 * Runtime configuration for the inference service.
 *
 * <p>Values are resolved (in order of precedence):
 * <ol>
 *   <li>Environment variables ({@code ONNX_MODEL_PATH}, {@code FEAST_BASE_URL}) —
 *       useful when running the service in a container next to the Feast container.</li>
 *   <li>The properties file {@code inference.properties} (classpath, or a path
 *       given to {@link #load(Path)}).</li>
 *   <li>Built-in defaults.</li>
 * </ol>
 */
public class InferenceConfig {

    public static final String DEFAULT_RESOURCE = "inference.properties";

    private static final String KEY_ONNX_PATH = "onnx.model.path";
    private static final String KEY_FEAST_URL = "feast.base.url";

    private static final String ENV_ONNX_PATH = "ONNX_MODEL_PATH";
    private static final String ENV_FEAST_URL = "FEAST_BASE_URL";

    private static final String DEFAULT_ONNX_PATH = "./model.onnx";
    private static final String DEFAULT_FEAST_URL = "http://localhost:6566";

    private final String onnxModelPath;
    private final String feastBaseUrl;

    public InferenceConfig(String onnxModelPath, String feastBaseUrl) {
        this.onnxModelPath = onnxModelPath;
        this.feastBaseUrl = feastBaseUrl;
    }

    public String getOnnxModelPath() {
        return onnxModelPath;
    }

    public String getFeastBaseUrl() {
        return feastBaseUrl;
    }

    /** Loads config from the {@code inference.properties} classpath resource + env overrides. */
    public static InferenceConfig load() {
        Properties props = new Properties();
        try (InputStream in = InferenceConfig.class.getClassLoader()
                .getResourceAsStream(DEFAULT_RESOURCE)) {
            if (in != null) {
                props.load(in);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read " + DEFAULT_RESOURCE, e);
        }
        return fromProperties(props);
    }

    /** Loads config from an explicit properties file path + env overrides. */
    public static InferenceConfig load(Path propertiesFile) {
        Properties props = new Properties();
        try (InputStream in = Files.newInputStream(propertiesFile)) {
            props.load(in);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read " + propertiesFile, e);
        }
        return fromProperties(props);
    }

    public static InferenceConfig fromProperties(Properties props) {
        String onnx = resolve(ENV_ONNX_PATH, props, KEY_ONNX_PATH, DEFAULT_ONNX_PATH);
        String feast = resolve(ENV_FEAST_URL, props, KEY_FEAST_URL, DEFAULT_FEAST_URL);
        return new InferenceConfig(onnx, feast);
    }

    private static String resolve(String envKey, Properties props, String propKey, String fallback) {
        String env = System.getenv(envKey);
        if (env != null && !env.isBlank()) {
            return env;
        }
        return props.getProperty(propKey, fallback);
    }

    @Override
    public String toString() {
        return "InferenceConfig{onnxModelPath='" + onnxModelPath
                + "', feastBaseUrl='" + feastBaseUrl + "'}";
    }
}
