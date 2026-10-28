package com.project.mlops.web;

import com.project.mlops.scraper.CrawlerBlockedException;
import com.project.mlops.scraper.CrawlerException;
import com.project.mlops.scraper.InvalidCrawlInputException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

/**
 * Maps errors to RFC 7807 JSON bodies. Spring MVC's own client errors (unreadable JSON, missing multipart part,
 * unknown enum value in a query parameter, ...) become 400 through {@link ResponseEntityExceptionHandler}.
 */
@RestControllerAdvice
public class ApiExceptionHandler extends ResponseEntityExceptionHandler {

    @ExceptionHandler(InvalidCrawlInputException.class)
    public ProblemDetail handleInvalidInput(InvalidCrawlInputException e) {
        return problem(HttpStatus.BAD_REQUEST, "Invalid input", e.getMessage());
    }

    @ExceptionHandler(CrawlerBlockedException.class)
    public ProblemDetail handleBlocked(CrawlerBlockedException e) {
        return problem(HttpStatus.SERVICE_UNAVAILABLE, "Blocked by Amazon", e.getMessage());
    }

    @ExceptionHandler(CrawlerException.class)
    public ProblemDetail handleCrawlerFailure(CrawlerException e) {
        return problem(HttpStatus.BAD_GATEWAY, "Crawl failed", e.getMessage());
    }

    private static ProblemDetail problem(HttpStatus status, String title, String detail) {
        ProblemDetail problem = ProblemDetail.forStatusAndDetail(status, detail);
        problem.setTitle(title);
        return problem;
    }
}
