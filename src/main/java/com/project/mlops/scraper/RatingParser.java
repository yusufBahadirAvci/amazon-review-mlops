package com.project.mlops.scraper;

import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses Amazon star-rating labels into a whole number.
 *
 * <p>Supported formats: {@code "4.0 out of 5 stars"} (English) and {@code "5 yıldız üzerinden 4,0"} (Turkish).
 * The result is not range-checked; callers decide what counts as a valid rating.
 */
public final class RatingParser {

    private static final String NUMBER = "(\\d+(?:[.,]\\d+)?)";

    private static final Pattern ENGLISH = Pattern.compile(NUMBER + "\\s+out\\s+of\\s+5\\b",
            Pattern.CASE_INSENSITIVE);

    // "5 yıldız üzerinden 4,0" - the rating comes after the scale
    private static final Pattern TURKISH = Pattern.compile("5\\s+yıldız\\s+üzerinden\\s+" + NUMBER,
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);

    private RatingParser() {
    }

    public static OptionalInt parse(String label) {
        if (label == null) {
            return OptionalInt.empty();
        }
        String normalized = label.replace(' ', ' ');

        Matcher english = ENGLISH.matcher(normalized);
        if (english.find()) {
            return toRating(english.group(1));
        }
        Matcher turkish = TURKISH.matcher(normalized);
        if (turkish.find()) {
            return toRating(turkish.group(1));
        }
        return OptionalInt.empty();
    }

    private static OptionalInt toRating(String number) {
        double value = Double.parseDouble(number.replace(',', '.'));
        return OptionalInt.of((int) Math.round(value));
    }
}
