package com.project.mlops.review;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ReviewRepository extends JpaRepository<Review, Long> {

    boolean existsByContentHash(String contentHash);

    List<Review> findBySource(ReviewSource source);
}
