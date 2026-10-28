package com.project.mlops;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class ProductReviewClassifierApplication {

    public static void main(String[] args) {
        SpringApplication.run(ProductReviewClassifierApplication.class, args);
    }
}
