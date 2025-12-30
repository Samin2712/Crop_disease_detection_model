package com.example.agriverse.dto;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class CreateTopicRequest {
    private String name;
    private String description;
}
