package com.example.agriverse.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter @Setter @AllArgsConstructor @NoArgsConstructor
public class SigninRequest {

    @NotEmpty(message = "Name field cannot be empty!\n")
    @NotBlank
    private String email;

    @NotEmpty(message = "Password cannot be empty!\n")
    @NotBlank
    private String password;

}
