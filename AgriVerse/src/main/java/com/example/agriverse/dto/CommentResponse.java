package com.example.agriverse.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.Instant;

@Getter
@Builder
public class CommentResponse {
    private Long id;
    private Long postId;
    private String authorUsername;
    private String content;
    private Instant createdAt;
}
