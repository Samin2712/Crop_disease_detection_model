package com.example.agriverse.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import lombok.Getter;

@Getter
public class SignupRequest {

    @NotEmpty(message = "Username cannot be empty!\n")
    @NotBlank
    private String username;

    @NotEmpty(message = "Email cannot be empty!\n")
    @NotBlank
    private String email;

    @NotEmpty(message = "Password cannot be empty!\n")
    @NotBlank
    private String password;

    // NEW: "USER" | "GOVT_OFFICER"
    private String accountType;

    // NEW: required if accountType == "GOVT_OFFICER"
    private String identificationNumber;
}
