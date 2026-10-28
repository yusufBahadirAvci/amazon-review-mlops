package com.project.mlops.review;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;

/**
 * A single product review. Reviewer names are deliberately not stored.
 */
@Entity
@Table(name = "reviews")
public class Review {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "product_id", length = 20)
    private String productId;

    @Column(columnDefinition = "TEXT")
    private String title;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String text;

    @Column(nullable = false)
    private int rating;

    @Column(name = "review_date")
    private String reviewDate;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 32)
    private ReviewSource source;

    @Column(name = "source_url", length = 2048)
    private String sourceUrl;

    @Column(name = "content_hash", nullable = false, unique = true, length = 64)
    private String contentHash;

    @Column(name = "crawled_at", nullable = false)
    private Instant crawledAt;

    protected Review() {
        // Required by JPA
    }

    public Review(String productId, String title, String text, int rating, String reviewDate,
                  ReviewSource source, String sourceUrl, String contentHash, Instant crawledAt) {
        this.productId = productId;
        this.title = title;
        this.text = text;
        this.rating = rating;
        this.reviewDate = reviewDate;
        this.source = source;
        this.sourceUrl = sourceUrl;
        this.contentHash = contentHash;
        this.crawledAt = crawledAt;
    }

    public Long getId() {
        return id;
    }

    public String getProductId() {
        return productId;
    }

    public String getTitle() {
        return title;
    }

    public String getText() {
        return text;
    }

    public int getRating() {
        return rating;
    }

    public String getReviewDate() {
        return reviewDate;
    }

    public ReviewSource getSource() {
        return source;
    }

    public String getSourceUrl() {
        return sourceUrl;
    }

    public String getContentHash() {
        return contentHash;
    }

    public Instant getCrawledAt() {
        return crawledAt;
    }
}
