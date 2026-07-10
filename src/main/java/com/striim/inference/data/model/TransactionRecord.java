package com.striim.inference.data.model;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Date;

public class TransactionRecord {

    private long id;
    private String status;
    private Date timestamp;
    private float value;
    private long custId;
    private long deviceId;
    private long merchantId;

    private static int TOTAL_ARGS = 7;

    public TransactionRecord(long id, String status, Date timestamp,
                             float value,
                             long custId, long deviceId, long merchantId) {
        this.id = id;
        this.status = status;
        this.timestamp = timestamp;
        this.value = value;
        this.custId = custId;
        this.deviceId = deviceId;
        this.merchantId = merchantId;
    }

    public static TransactionRecord parse(Object... inputs){
        if(inputs.length < TOTAL_ARGS)
            throw new IllegalArgumentException("Number of inputs expected::"
                    + TOTAL_ARGS +
                    ", but actual received::" + inputs.length);
        long id = Long.parseLong(inputs[0].toString());
        String status = inputs[1].toString();
        Date timestamp = parseDate(inputs[2].toString());
        float value = Float.parseFloat(inputs[3].toString());

        long custId = Long.parseLong(inputs[4].toString());
        long deviceId = Long.parseLong(inputs[5].toString());
        long merchantId = Long.parseLong(inputs[6].toString());

        return new TransactionRecord(id, status, timestamp,
                value, custId, deviceId, merchantId);
    }

    public long getId() {
        return id;
    }

    public String getStatus() {
        return status;
    }

    public Date getTimestamp() {
        return timestamp;
    }

    public float getValue() {
        return value;
    }

    public long getCustId() {
        return custId;
    }

    public long getDeviceId() {
        return deviceId;
    }

    public long getMerchantId() {
        return merchantId;
    }

    // Space-separated local timestamps, with or without fractional seconds
    // (e.g. "2024-10-07 19:18:33.228" or "2024-10-07 19:18:33").
    private static final DateTimeFormatter[] LOCAL_FORMATS = {
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
    };

    /**
     * Parses a transaction timestamp, accepting several common shapes:
     * <ul>
     *   <li>ISO-8601 instant with a zone/offset, e.g. {@code 2025-01-24T12:15:47.485Z}
     *       or {@code 2025-01-24T12:15:47+05:30};</li>
     *   <li>ISO-8601 local date-time, e.g. {@code 2025-01-24T12:15:47.485};</li>
     *   <li>space-separated local date-time, e.g. {@code 2024-10-07 19:18:33.228}.</li>
     * </ul>
     */
    private static Date parseDate(String dateStr) {
        String s = dateStr.trim();

        // Zoned/offset instant (has 'Z' or an explicit offset) -> absolute time.
        try {
            return Date.from(Instant.parse(s));
        } catch (DateTimeParseException ignored) {
            // not a plain instant
        }
        try {
            return Date.from(OffsetDateTime.parse(s).toInstant());
        } catch (DateTimeParseException ignored) {
            // not an offset date-time
        }

        // ISO-8601 local date-time (the 'T' separator, no zone).
        try {
            return toDate(LocalDateTime.parse(s));
        } catch (DateTimeParseException ignored) {
            // not ISO local
        }

        // Space-separated local formats.
        for (DateTimeFormatter formatter : LOCAL_FORMATS) {
            try {
                return toDate(LocalDateTime.parse(s, formatter));
            } catch (DateTimeParseException ignored) {
                // try the next pattern
            }
        }

        throw new IllegalArgumentException("Unparseable transaction timestamp: '" + dateStr + "'");
    }

    private static Date toDate(LocalDateTime localDateTime) {
        return Date.from(localDateTime.atZone(ZoneId.systemDefault()).toInstant());
    }
}
