package com.example.agriverse.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class TopicResponse {
    private Long id;
    private String name;
    private String description;
    private long postCount;
}
