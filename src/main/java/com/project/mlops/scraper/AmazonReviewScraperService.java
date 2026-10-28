package com.project.mlops.scraper;

import com.project.mlops.review.Review;
import com.project.mlops.review.ReviewRepository;
import com.project.mlops.review.ReviewSource;
import org.jsoup.Connection;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Crawls Amazon review pages, either live or from saved HTML files, and stores the reviews.
 *
 * <p>The crawler sends one configurable User-Agent, waits {@code crawler.delay-ms} between live requests and
 * stops when Amazon blocks it; it does not try to evade bot protection.
 */
@Service
public class AmazonReviewScraperService {

    private static final Logger log = LoggerFactory.getLogger(AmazonReviewScraperService.class);

    private static final Pattern AMAZON_HOST =
            Pattern.compile("^(?:[a-z0-9-]+\\.)*amazon\\.(?:com|[a-z]{2}|co\\.[a-z]{2}|com\\.[a-z]{2})$");
    private static final Pattern ASIN =
            Pattern.compile("/(?:dp|gp/product|product-reviews|gp/aw/d)/([A-Z0-9]{10})(?=[/?#]|$)");
    private static final Set<Integer> BLOCKED_STATUSES = Set.of(403, 503);
    // Markup inside a review body that is UI chrome rather than review text
    private static final String BODY_NOISE = "script, style, noscript, .a-expander-header";

    private final CrawlerProperties properties;
    private final ReviewRepository reviewRepository;

    private final Object throttleLock = new Object();
    private long lastRequestAtMillis;

    public AmazonReviewScraperService(CrawlerProperties properties, ReviewRepository reviewRepository) {
        this.properties = properties;
        this.reviewRepository = reviewRepository;
    }

    /**
     * Fetches an Amazon review page and stores its reviews.
     *
     * @throws InvalidCrawlInputException if the URL is not an http(s) Amazon URL
     * @throws CrawlerBlockedException    if Amazon answers 403/503 or shows a captcha
     * @throws CrawlerException           if the page cannot be fetched for another reason
     */
    public CrawlResult crawlUrl(String url) {
        URI uri = validateAmazonUrl(url);
        Document document = fetch(uri);
        return parseAndSave(document, uri.toString(), ReviewSource.CRAWLER_LIVE);
    }

    /**
     * Stores the reviews from a review page saved in a browser ("Save page as...").
     *
     * @param sourceUrl original page URL, optional; used for the product ID (ASIN)
     * @throws InvalidCrawlInputException if the HTML is empty
     * @throws CrawlerBlockedException    if the saved page is a captcha page
     */
    public CrawlResult importHtml(String html, String sourceUrl) {
        if (html == null || html.isBlank()) {
            throw new InvalidCrawlInputException("HTML content is empty");
        }
        String url = sourceUrl == null || sourceUrl.isBlank() ? null : sourceUrl.trim();
        Document document = Jsoup.parse(html, url == null ? "" : url);
        return parseAndSave(document, url, ReviewSource.CRAWLER_HTML_FILE);
    }

    static boolean isBlockedPage(Document document) {
        if (document.selectFirst("form[action*=validateCaptcha], input#captchacharacters") != null) {
            return true;
        }
        if (document.title().toLowerCase(Locale.ROOT).contains("robot check")) {
            return true;
        }
        String text = document.text().toLowerCase(Locale.ROOT);
        return text.contains("enter the characters you see below")
                || text.contains("to discuss automated access to amazon data");
    }

    static String extractProductId(String url) {
        if (url == null) {
            return null;
        }
        Matcher matcher = ASIN.matcher(url);
        return matcher.find() ? matcher.group(1) : null;
    }

    static String contentHash(String productId, String text) {
        String normalized = text.toLowerCase(Locale.ROOT).replaceAll("\\s+", " ").trim();
        String input = (productId == null ? "" : productId) + "|" + normalized;
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(input.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private CrawlResult parseAndSave(Document document, String sourceUrl, ReviewSource source) {
        if (isBlockedPage(document)) {
            throw new CrawlerBlockedException("Amazon returned a captcha / robot-check page"
                    + (sourceUrl == null ? "" : " for " + sourceUrl));
        }

        List<String> warnings = new ArrayList<>();
        Elements elements = document.select(properties.selectors().review());
        int found = elements.size();
        if (found == 0) {
            warnings.add(noReviewsWarning(document));
        }

        List<Element> toProcess = elements;
        if (found > properties.maxReviewsPerRun()) {
            toProcess = elements.subList(0, properties.maxReviewsPerRun());
            warnings.add("Only the first " + properties.maxReviewsPerRun() + " of " + found
                    + " reviews were processed (crawler.max-reviews-per-run)");
        }

        String productId = extractProductId(sourceUrl);
        Instant crawledAt = Instant.now();
        Set<String> seenHashes = new HashSet<>();
        int saved = 0;
        int skippedInvalid = 0;
        int skippedDuplicate = 0;

        for (int i = 0; i < toProcess.size(); i++) {
            ParsedReview parsed = parse(toProcess.get(i));
            String problem = validate(parsed);
            if (problem != null) {
                skippedInvalid++;
                warnings.add("Review " + (i + 1) + " skipped: " + problem);
                continue;
            }

            String hash = contentHash(productId, parsed.text());
            if (!seenHashes.add(hash) || reviewRepository.existsByContentHash(hash)) {
                skippedDuplicate++;
                continue;
            }
            try {
                reviewRepository.save(new Review(productId, parsed.title(), parsed.text(), parsed.rating(),
                        parsed.reviewDate(), source, sourceUrl, hash, crawledAt));
                saved++;
            } catch (DataIntegrityViolationException e) {
                // Another import stored the same review in the meantime
                skippedDuplicate++;
            }
        }

        log.info("{} {}: found={}, saved={}, skippedInvalid={}, skippedDuplicate={}", source,
                sourceUrl == null ? "(no URL)" : sourceUrl, found, saved, skippedInvalid, skippedDuplicate);
        return new CrawlResult(found, saved, skippedInvalid, skippedDuplicate, warnings);
    }

    private ParsedReview parse(Element review) {
        CrawlerProperties.Selectors selectors = properties.selectors();

        Integer rating = null;
        Element ratingElement = review.selectFirst(selectors.rating());
        if (ratingElement != null) {
            OptionalInt parsed = RatingParser.parse(ratingElement.text());
            rating = parsed.isPresent() ? parsed.getAsInt() : null;
        }

        String title = null;
        Element titleElement = review.selectFirst(selectors.title());
        if (titleElement != null) {
            // Amazon nests the star label inside the title link
            Element copy = titleElement.clone();
            copy.select(selectors.rating()).remove();
            title = blankToNull(clean(copy.text()));
        }

        String text = "";
        Element bodyElement = review.selectFirst(selectors.text());
        if (bodyElement != null) {
            Element copy = bodyElement.clone();
            copy.select(BODY_NOISE).remove();
            text = clean(copy.text());
        }

        Element dateElement = review.selectFirst(selectors.date());
        String reviewDate = dateElement == null ? null : blankToNull(clean(dateElement.text()));

        return new ParsedReview(title, text, rating, reviewDate);
    }

    private static String validate(ParsedReview review) {
        if (review.text().isEmpty()) {
            return "review text is empty";
        }
        if (review.rating() == null) {
            return "star rating is missing or has an unknown format";
        }
        if (review.rating() < 1 || review.rating() > 5) {
            return "rating " + review.rating() + " is outside 1-5";
        }
        return null;
    }

    private String noReviewsWarning(Document document) {
        boolean signIn = document.selectFirst("form[name=signIn]") != null
                || document.title().toLowerCase(Locale.ROOT).contains("sign-in");
        if (signIn) {
            return "Amazon returned a sign-in page; review pages may require a logged-in session";
        }
        return "No elements matched '" + properties.selectors().review() + "'; the page layout may have changed";
    }

    private Document fetch(URI uri) {
        throttle();
        Connection.Response response;
        try {
            response = Jsoup.connect(uri.toString())
                    .userAgent(properties.userAgent())
                    .timeout(properties.timeoutMs())
                    .followRedirects(true)
                    .ignoreHttpErrors(true)
                    .execute();
        } catch (IOException e) {
            throw new CrawlerException("Failed to fetch " + uri + ": " + e.getMessage(), e);
        }

        int status = response.statusCode();
        if (BLOCKED_STATUSES.contains(status)) {
            throw new CrawlerBlockedException("Amazon responded with HTTP " + status + " for " + uri);
        }
        if (status >= 400) {
            throw new CrawlerException("Amazon responded with HTTP " + status + " for " + uri);
        }
        String finalHost = response.url().getHost();
        if (!isAmazonHost(finalHost)) {
            throw new CrawlerException("Request for " + uri + " was redirected to non-Amazon host " + finalHost);
        }
        try {
            return response.parse();
        } catch (IOException e) {
            throw new CrawlerException("Failed to read the response from " + uri + ": " + e.getMessage(), e);
        }
    }

    // Keeps at least crawler.delay-ms between two live requests, across all callers
    private void throttle() {
        synchronized (throttleLock) {
            long waitMillis = lastRequestAtMillis + properties.delayMs() - System.currentTimeMillis();
            if (waitMillis > 0) {
                try {
                    Thread.sleep(waitMillis);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new CrawlerException("Interrupted while waiting between requests", e);
                }
            }
            lastRequestAtMillis = System.currentTimeMillis();
        }
    }

    private static URI validateAmazonUrl(String url) {
        if (url == null || url.isBlank()) {
            throw new InvalidCrawlInputException("url must not be blank");
        }
        URI uri;
        try {
            uri = new URI(url.trim());
        } catch (URISyntaxException e) {
            throw new InvalidCrawlInputException("url is not a valid URL: " + url);
        }
        String scheme = uri.getScheme();
        if (scheme == null || !(scheme.equalsIgnoreCase("http") || scheme.equalsIgnoreCase("https"))) {
            throw new InvalidCrawlInputException("url must start with http:// or https://");
        }
        if (!isAmazonHost(uri.getHost())) {
            throw new InvalidCrawlInputException("url must point to an Amazon site, e.g. https://www.amazon.com/...");
        }
        return uri;
    }

    private static boolean isAmazonHost(String host) {
        return host != null && AMAZON_HOST.matcher(host.toLowerCase(Locale.ROOT)).matches();
    }

    private static String clean(String raw) {
        return raw.replace(' ', ' ').replaceAll("\\s+", " ").trim();
    }

    private static String blankToNull(String value) {
        return value.isEmpty() ? null : value;
    }

    private record ParsedReview(String title, String text, Integer rating, String reviewDate) {
    }
}
