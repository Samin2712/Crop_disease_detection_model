package com.example.agriverse.controller;

import com.example.agriverse.dto.UserInfo;
import com.example.agriverse.model.Role;
import com.example.agriverse.model.User;
import com.example.agriverse.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/util")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class UtilityController {
    private final UserRepository userRepository;

    @PreAuthorize("isAuthenticated()")
    @GetMapping("/user-info/{username}")
    public ResponseEntity<?> userInfo(@PathVariable String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("Error!"));
        UserInfo userInfo = new UserInfo();

        userInfo.setUsername(user.getUsername());
        userInfo.setEmail(user.getEmail());
        userInfo.setIdentificationNumber(user.getIdentificationNumber());
        return ResponseEntity.ok(userInfo);
    }

    @PreAuthorize("hasRole('ADMIN')")
    @PostMapping("/grant-admin-access")
    public ResponseEntity<?> adminAccess(@RequestBody String username)
    {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("Error!"));
        user.getRoles().add(new Role("ROLE_ADMIN"));

        return ResponseEntity.ok("Admin added");
    }

}
