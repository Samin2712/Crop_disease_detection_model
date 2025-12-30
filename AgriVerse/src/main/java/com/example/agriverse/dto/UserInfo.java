package com.example.agriverse.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Set;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class UserInfo {

    private String username;
    private String email;

    // Visible only to govt officer / admin
    private String identificationNumber;

    // ROLE_USER, ROLE_GOVT_OFFICER, ROLE_ADMIN
    private Set<String> roles;
}
