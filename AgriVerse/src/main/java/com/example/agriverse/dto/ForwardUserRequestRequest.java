package com.example.agriverse.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ForwardUserRequestRequest {
    private String toOfficerUsername; // or ID, but username is simplest
    private String note; // optional
}
