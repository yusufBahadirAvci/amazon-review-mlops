package com.project.mlops.review;

import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/reviews")
public class ReviewController {

    private final ReviewRepository reviewRepository;

    public ReviewController(ReviewRepository reviewRepository) {
        this.reviewRepository = reviewRepository;
    }

    @GetMapping
    public List<ReviewResponse> list(@RequestParam(required = false) ReviewSource source) {
        List<Review> reviews = source == null
                ? reviewRepository.findAll(Sort.by("id"))
                : reviewRepository.findBySource(source);
        return reviews.stream().map(ReviewResponse::from).toList();
    }
}
