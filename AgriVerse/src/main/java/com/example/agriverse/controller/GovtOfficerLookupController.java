package com.example.agriverse.controller;

import com.example.agriverse.model.User;
import com.example.agriverse.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class GovtOfficerLookupController {

    private final UserRepository userRepo;

    public record OfficerInfo(String username, String identificationNumber) {}

    @PreAuthorize("hasAnyRole('GOVT_OFFICER','ADMIN')")
    @GetMapping("/govt-officers")
    public List<OfficerInfo> listGovtOfficers() {
        return userRepo.findByRoles_Name("ROLE_GOVT_OFFICER").stream()
                .map(u -> new OfficerInfo(u.getUsername(), u.getIdentificationNumber()))
                .collect(Collectors.toList());
    }
}
