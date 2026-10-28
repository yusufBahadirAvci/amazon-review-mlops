package com.project.mlops.review;

import java.time.Instant;

/**
 * API view of a stored review (the content hash is an internal detail).
 */
public record ReviewResponse(Long id, String productId, String title, String text, int rating,
                             String reviewDate, ReviewSource source, String sourceUrl, Instant crawledAt) {

    public static ReviewResponse from(Review review) {
        return new ReviewResponse(review.getId(), review.getProductId(), review.getTitle(), review.getText(),
                review.getRating(), review.getReviewDate(), review.getSource(), review.getSourceUrl(),
                review.getCrawledAt());
    }
}
