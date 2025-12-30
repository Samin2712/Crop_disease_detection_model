package com.example.agriverse.dto;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class CreatePostRequest {
    private Long topicId;
    private String title;
    private String content;
}
