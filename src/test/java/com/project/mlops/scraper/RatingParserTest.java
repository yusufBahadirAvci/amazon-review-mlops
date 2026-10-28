package com.project.mlops.scraper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.OptionalInt;

import static org.assertj.core.api.Assertions.assertThat;

class RatingParserTest {

    @ParameterizedTest
    @CsvSource(delimiter = '|', value = {
        "4.0 out of 5 stars | 4",
        "1.0 out of 5 stars | 1",
        "5 out of 5 stars | 5",
        "  3.0  OUT OF 5 STARS  | 3",
        "4.6 out of 5 stars | 5"
    })
    void parsesEnglishFormat(String label, int expected) {
        assertThat(RatingParser.parse(label)).hasValue(expected);
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', value = {
        "5 yıldız üzerinden 4,0 | 4",
        "5 yıldız üzerinden 1,0 | 1",
        "5 Yıldız üzerinden 5 | 5",
        "5 yıldız üzerinden 2.0 | 2"
    })
    void parsesTurkishFormat(String label, int expected) {
        assertThat(RatingParser.parse(label)).hasValue(expected);
    }

    @Test
    void takesTheRatingNotTheScaleInTurkishFormat() {
        assertThat(RatingParser.parse("5 yıldız üzerinden 2,0")).hasValue(2);
    }

    @Test
    void handlesNonBreakingSpaces() {
        assertThat(RatingParser.parse("4,0 out of 5 stars")).hasValue(4);
        assertThat(RatingParser.parse("5 yıldız üzerinden 3,0")).hasValue(3);
    }

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {"great product", "4 stars", "4,0 von 5 Sternen", "out of 5 stars"})
    void returnsEmptyForUnknownFormats(String label) {
        assertThat(RatingParser.parse(label)).isEqualTo(OptionalInt.empty());
    }
}
