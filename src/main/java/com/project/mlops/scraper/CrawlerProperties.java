package com.project.mlops.scraper;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Crawler settings bound from the {@code crawler.*} keys in application.properties.
 *
 * @param userAgent        User-Agent header sent on live requests
 * @param timeoutMs        connect/read timeout for live requests
 * @param delayMs          minimum pause between two live requests
 * @param maxReviewsPerRun upper limit of reviews processed per crawl or import
 * @param selectors        CSS selectors used to parse a review page
 */
@ConfigurationProperties(prefix = "crawler")
public record CrawlerProperties(String userAgent, int timeoutMs, long delayMs, int maxReviewsPerRun,
                                Selectors selectors) {

    /**
     * CSS selectors; {@code review} is applied to the page, the others inside each review element.
     */
    public record Selectors(String review, String rating, String title, String text, String date) {
    }
}
