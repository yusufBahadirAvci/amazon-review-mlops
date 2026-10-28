package com.project.mlops.scraper;

/**
 * Amazon refused the request (HTTP 403/503) or returned a captcha / robot-check page.
 * The crawler does not try to work around this.
 */
public class CrawlerBlockedException extends CrawlerException {

    public CrawlerBlockedException(String message) {
        super(message);
    }
}
