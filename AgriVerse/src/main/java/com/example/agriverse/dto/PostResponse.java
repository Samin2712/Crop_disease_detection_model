package com.example.agriverse.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.Instant;

@Getter
@Builder
public class PostResponse {
    private Long id;
    private Long topicId;
    private String topicName;
    private String title;
    private String content;
    private String authorUsername;
    private Instant createdAt;
    private Instant updatedAt;
    private long commentCount;
}
