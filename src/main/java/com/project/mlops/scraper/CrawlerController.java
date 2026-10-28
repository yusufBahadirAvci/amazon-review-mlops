package com.project.mlops.scraper;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

@RestController
@RequestMapping("/api/crawler")
public class CrawlerController {

    private final AmazonReviewScraperService scraperService;

    public CrawlerController(AmazonReviewScraperService scraperService) {
        this.scraperService = scraperService;
    }

    @PostMapping(path = "/url", consumes = MediaType.APPLICATION_JSON_VALUE)
    public CrawlResult crawlUrl(@RequestBody CrawlUrlRequest request) {
        return scraperService.crawlUrl(request.url());
    }

    @PostMapping(path = "/html", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public CrawlResult importHtml(@RequestPart("file") MultipartFile file,
                                  @RequestParam(name = "sourceUrl", required = false) String sourceUrl)
            throws IOException {
        if (file.isEmpty()) {
            throw new InvalidCrawlInputException("Uploaded file is empty");
        }
        String html = new String(file.getBytes(), StandardCharsets.UTF_8);
        return scraperService.importHtml(html, sourceUrl);
    }

    public record CrawlUrlRequest(String url) {
    }
}
