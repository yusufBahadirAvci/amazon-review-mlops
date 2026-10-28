package com.project.mlops.scraper;

/**
 * The caller supplied an unusable URL or HTML document.
 */
public class InvalidCrawlInputException extends RuntimeException {

    public InvalidCrawlInputException(String message) {
        super(message);
    }
}
