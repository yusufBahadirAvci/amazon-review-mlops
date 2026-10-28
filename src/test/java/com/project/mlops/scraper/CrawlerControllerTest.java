package com.project.mlops.scraper;

import com.project.mlops.review.ReviewRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import java.io.IOException;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class CrawlerControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ReviewRepository reviewRepository;

    @BeforeEach
    void cleanDatabase() {
        reviewRepository.deleteAll();
    }

    @Test
    void htmlUploadImportsReviewsAndListsThemBySource() throws Exception {
        mockMvc.perform(multipart("/api/crawler/html")
                        .file(htmlFile("amazon-reviews.html"))
                        .param("sourceUrl", "https://www.amazon.com/dp/B0TEST1234"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.found").value(3))
                .andExpect(jsonPath("$.saved").value(2))
                .andExpect(jsonPath("$.skippedInvalid").value(1))
                .andExpect(jsonPath("$.skippedDuplicate").value(0));

        mockMvc.perform(get("/api/reviews").param("source", "CRAWLER_HTML_FILE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2))
                .andExpect(jsonPath("$[0].productId").value("B0TEST1234"))
                .andExpect(jsonPath("$[0].contentHash").doesNotExist());

        mockMvc.perform(get("/api/reviews").param("source", "DATASET"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(0));
    }

    @Test
    void captchaUploadReturns503() throws Exception {
        mockMvc.perform(multipart("/api/crawler/html").file(htmlFile("amazon-captcha.html")))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.status").value(503))
                .andExpect(jsonPath("$.detail").exists());
    }

    @Test
    void emptyUploadReturns400() throws Exception {
        MockMultipartFile empty = new MockMultipartFile("file", "empty.html", MediaType.TEXT_HTML_VALUE, new byte[0]);

        mockMvc.perform(multipart("/api/crawler/html").file(empty))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.detail").value("Uploaded file is empty"));
    }

    @Test
    void missingFilePartReturns400() throws Exception {
        mockMvc.perform(multipart("/api/crawler/html"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400));
    }

    @Test
    void nonAmazonUrlReturns400() throws Exception {
        mockMvc.perform(post("/api/crawler/url")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"url\": \"https://example.com/dp/B0TEST1234\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.detail").exists());
    }

    @Test
    void malformedJsonReturns400() throws Exception {
        mockMvc.perform(post("/api/crawler/url")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{not json"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400));
    }

    @Test
    void unknownSourceReturns400() throws Exception {
        mockMvc.perform(get("/api/reviews").param("source", "NOT_A_SOURCE"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400));
    }

    private static MockMultipartFile htmlFile(String fixture) throws IOException {
        byte[] content = new ClassPathResource("fixtures/" + fixture).getContentAsByteArray();
        return new MockMultipartFile("file", fixture, MediaType.TEXT_HTML_VALUE, content);
    }
}
