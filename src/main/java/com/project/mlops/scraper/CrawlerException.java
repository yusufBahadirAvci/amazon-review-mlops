package com.project.mlops.scraper;

/**
 * The target page could not be fetched or read.
 */
public class CrawlerException extends RuntimeException {

    public CrawlerException(String message) {
        super(message);
    }

    public CrawlerException(String message, Throwable cause) {
        super(message, cause);
    }
}
