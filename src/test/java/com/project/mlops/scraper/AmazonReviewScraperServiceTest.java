package com.project.mlops.scraper;

import com.project.mlops.review.Review;
import com.project.mlops.review.ReviewRepository;
import com.project.mlops.review.ReviewSource;
import org.jsoup.Jsoup;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
class AmazonReviewScraperServiceTest {

    private static final String SOURCE_URL = "https://www.amazon.com/Test-Headphones/product-reviews/B0TEST1234/ref=cm";

    @Autowired
    private AmazonReviewScraperService scraperService;

    @Autowired
    private ReviewRepository reviewRepository;

    @BeforeEach
    void cleanDatabase() {
        reviewRepository.deleteAll();
    }

    @Test
    void importsValidReviewsAndSkipsEmptyBody() throws IOException {
        CrawlResult result = scraperService.importHtml(fixture("amazon-reviews.html"), SOURCE_URL);

        assertThat(result.found()).isEqualTo(3);
        assertThat(result.saved()).isEqualTo(2);
        assertThat(result.skippedInvalid()).isEqualTo(1);
        assertThat(result.skippedDuplicate()).isZero();
        assertThat(result.warnings()).containsExactly("Review 3 skipped: review text is empty");

        List<Review> stored = reviewRepository.findBySource(ReviewSource.CRAWLER_HTML_FILE).stream()
                .sorted(Comparator.comparing(Review::getId))
                .toList();
        assertThat(stored).hasSize(2);

        Review english = stored.get(0);
        assertThat(english.getProductId()).isEqualTo("B0TEST1234");
        assertThat(english.getTitle()).isEqualTo("Solid build, average battery");
        assertThat(english.getText()).isEqualTo("Works well for daily use. Battery lasts about a day.");
        assertThat(english.getRating()).isEqualTo(4);
        assertThat(english.getReviewDate()).isEqualTo("Reviewed in the United States on March 3, 2025");
        assertThat(english.getSourceUrl()).isEqualTo(SOURCE_URL);
        assertThat(english.getContentHash()).hasSize(64);
        assertThat(english.getCrawledAt()).isNotNull();

        Review turkish = stored.get(1);
        assertThat(turkish.getTitle()).isEqualTo("Fiyatına göre iyi");
        assertThat(turkish.getText()).isEqualTo("Kargo hızlıydı ama kutu biraz ezikti.");
        assertThat(turkish.getRating()).isEqualTo(3);
        assertThat(turkish.getReviewDate()).isEqualTo("Türkiye'de 12 Ocak 2025 tarihinde değerlendirildi");
    }

    @Test
    void secondImportOfSamePageSkipsDuplicates() throws IOException {
        String html = fixture("amazon-reviews.html");
        scraperService.importHtml(html, SOURCE_URL);

        CrawlResult second = scraperService.importHtml(html, SOURCE_URL);

        assertThat(second.found()).isEqualTo(3);
        assertThat(second.saved()).isZero();
        assertThat(second.skippedInvalid()).isEqualTo(1);
        assertThat(second.skippedDuplicate()).isEqualTo(2);
        assertThat(reviewRepository.count()).isEqualTo(2);
    }

    @Test
    void importWithoutSourceUrlStoresNullProductId() throws IOException {
        CrawlResult result = scraperService.importHtml(fixture("amazon-reviews.html"), null);

        assertThat(result.saved()).isEqualTo(2);
        assertThat(reviewRepository.findAll()).allSatisfy(review -> {
            assertThat(review.getProductId()).isNull();
            assertThat(review.getSourceUrl()).isNull();
        });
    }

    @Test
    void captchaPageIsDetected() throws IOException {
        String captcha = fixture("amazon-captcha.html");

        assertThat(AmazonReviewScraperService.isBlockedPage(Jsoup.parse(captcha))).isTrue();
        assertThatThrownBy(() -> scraperService.importHtml(captcha, SOURCE_URL))
                .isInstanceOf(CrawlerBlockedException.class)
                .hasMessageContaining("captcha");
        assertThat(reviewRepository.count()).isZero();
    }

    @Test
    void reviewPageIsNotMistakenForCaptcha() throws IOException {
        assertThat(AmazonReviewScraperService.isBlockedPage(Jsoup.parse(fixture("amazon-reviews.html")))).isFalse();
    }

    @Test
    void crawlUrlRejectsNonAmazonUrlsWithoutFetching() {
        assertThatThrownBy(() -> scraperService.crawlUrl("https://example.com/dp/B0TEST1234"))
                .isInstanceOf(InvalidCrawlInputException.class);
        assertThatThrownBy(() -> scraperService.crawlUrl("https://www.amazon.com.evil.net/dp/B0TEST1234"))
                .isInstanceOf(InvalidCrawlInputException.class);
        assertThatThrownBy(() -> scraperService.crawlUrl("ftp://www.amazon.com/dp/B0TEST1234"))
                .isInstanceOf(InvalidCrawlInputException.class);
        assertThatThrownBy(() -> scraperService.crawlUrl(" "))
                .isInstanceOf(InvalidCrawlInputException.class);
    }

    @Test
    void extractsAsinFromCommonUrlShapes() {
        assertThat(AmazonReviewScraperService.extractProductId("https://www.amazon.com/dp/B0TEST1234"))
                .isEqualTo("B0TEST1234");
        assertThat(AmazonReviewScraperService.extractProductId("https://www.amazon.com.tr/Urun/dp/B0TEST1234?th=1"))
                .isEqualTo("B0TEST1234");
        assertThat(AmazonReviewScraperService.extractProductId("https://www.amazon.de/gp/product/B0TEST1234/"))
                .isEqualTo("B0TEST1234");
        assertThat(AmazonReviewScraperService.extractProductId("https://www.amazon.com/s?k=headphones")).isNull();
    }

    @Test
    void contentHashIgnoresCaseAndWhitespace() {
        assertThat(AmazonReviewScraperService.contentHash("B0TEST1234", "Great  product\n"))
                .isEqualTo(AmazonReviewScraperService.contentHash("B0TEST1234", "great product"))
                .isNotEqualTo(AmazonReviewScraperService.contentHash("B0OTHER123", "great product"));
    }

    private static String fixture(String name) throws IOException {
        return new ClassPathResource("fixtures/" + name).getContentAsString(StandardCharsets.UTF_8);
    }
}
