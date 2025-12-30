package com.example.agriverse.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.Instant;

@Getter
@Builder
public class UserRequestMessageResponse {
    private Long id;
    private Long requestId;

    private String senderUsername;
    private String senderRole; // optional convenience for frontend

    private String message;
    private Instant createdAt;
}
