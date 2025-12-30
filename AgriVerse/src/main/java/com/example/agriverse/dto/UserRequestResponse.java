package com.example.agriverse.dto;

import com.example.agriverse.model.RequestStatus;
import lombok.Builder;
import lombok.Getter;

import java.time.Instant;

@Getter
@Builder
public class UserRequestResponse {
    private Long id;

    private String createdByUsername;

    private String category;
    private String description;
    private String imageUrl;

    private String state;
    private String district;

    private RequestStatus status;

    private Instant createdAt;
    private Instant takenAt;
    private Instant archivedAt;

    private String assignedOfficerUsername;
    private String assignedOfficerIdentificationNumber;

    // ✅ NEW: structured user details for chat UI
    private UserInfo createdBy;
    private UserInfo assignedOfficer;
}
