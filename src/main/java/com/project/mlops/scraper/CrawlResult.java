package com.project.mlops.scraper;

import java.util.List;

/**
 * Summary of one crawl or HTML import.
 *
 * @param found            review elements found on the page
 * @param saved            reviews stored in the database
 * @param skippedInvalid   reviews skipped because of blank text or a missing / out-of-range rating
 * @param skippedDuplicate reviews skipped because the same content is already stored
 * @param warnings         human-readable notes about skipped items or page problems
 */
public record CrawlResult(int found, int saved, int skippedInvalid, int skippedDuplicate, List<String> warnings) {

    public CrawlResult {
        warnings = List.copyOf(warnings);
    }
}
