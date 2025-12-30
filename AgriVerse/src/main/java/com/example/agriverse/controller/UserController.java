package com.example.agriverse.controller;

import com.example.agriverse.dto.SigninRequest;
import com.example.agriverse.dto.SignupRequest;
import com.example.agriverse.dto.ResendVerificationRequest;
import com.example.agriverse.model.Role;
import com.example.agriverse.model.User;
import com.example.agriverse.repository.RoleRepository;
import com.example.agriverse.repository.UserRepository;
import com.example.agriverse.security.AuthEntryPointJwt;
import com.example.agriverse.security.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Set;
import com.example.agriverse.model.VerificationToken;
import com.example.agriverse.repository.VerificationTokenRepository;
import com.example.agriverse.service.EmailService;

import java.time.Duration;
import java.time.Instant;
import java.util.UUID;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173") // frontend URL
public class UserController {

    private final UserRepository userRepo;
    private final RoleRepository roleRepo;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;
    private final AuthEntryPointJwt authEntryPointJwt;
    private final VerificationTokenRepository verificationTokenRepo;
    private final EmailService emailService;



    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody SignupRequest signupRequest) {

        String username = signupRequest.getUsername() == null ? "" : signupRequest.getUsername().trim();
        String email = signupRequest.getEmail() == null ? "" : signupRequest.getEmail().trim();

        if (username.isBlank()) return ResponseEntity.badRequest().body("Username is required!");
        if (email.isBlank()) return ResponseEntity.badRequest().body("Email is required!");

        if (userRepo.existsByUsername(username)) {
            return ResponseEntity.badRequest().body("Username already exists!");
        }
        if (userRepo.existsByEmail(email)) {
            return ResponseEntity.badRequest().body("Email already exists!");
        }

        String accountType = signupRequest.getAccountType();
        if (accountType == null || accountType.isBlank()) accountType = "USER";

        Role roleToAssign;

        if (accountType.equalsIgnoreCase("GOVT_OFFICER")) {
            String idNo = signupRequest.getIdentificationNumber();
            if (idNo == null || idNo.isBlank()) {
                return ResponseEntity.badRequest().body("Identification number is required for GOVT officer registration!");
            }
            idNo = idNo.trim();

            if (userRepo.existsByIdentificationNumber(idNo)) {
                return ResponseEntity.badRequest().body("Identification number already exists!");
            }

            roleToAssign = roleRepo.findByName("ROLE_GOVT_OFFICER")
                    .orElseThrow(() -> new RuntimeException("ROLE_GOVT_OFFICER not found"));
        } else {
            roleToAssign = roleRepo.findByName("ROLE_USER")
                    .orElseThrow(() -> new RuntimeException("ROLE_USER not found"));
        }

        User user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(signupRequest.getPassword()));
        user.setEmail(email);
        user.setRoles(Set.of(roleToAssign));
        user.setEmailVerified(false);

        if (accountType.equalsIgnoreCase("GOVT_OFFICER")) {
            user.setIdentificationNumber(signupRequest.getIdentificationNumber().trim());
        }

        User savedUser = userRepo.save(user);

        // email verification (unchanged)
        verificationTokenRepo.deleteByUser(savedUser);

        VerificationToken vt = new VerificationToken();
        vt.setToken(UUID.randomUUID().toString());
        vt.setUser(savedUser);
        vt.setExpiresAt(Instant.now().plus(Duration.ofHours(24)));
        verificationTokenRepo.save(vt);

        String verifyLink = "http://localhost:5173/verify-email?token=" + vt.getToken();

        emailService.send(
                savedUser.getEmail(),
                "Verify your AgriVerse account",
                "Click this link to verify your email:\n" + verifyLink + "\n\nThis link expires in 24 hours."
        );

        return ResponseEntity.ok(Map.of(
                "message", "Registration successful. Please verify your email before logging in.",
                "accountType", accountType
        ));
    }





    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody SigninRequest signinRequest) {

        User _user = userRepo.findByEmail(signinRequest.getEmail())
                .orElseThrow(() -> new RuntimeException("Invalid email or password"));

        if (!passwordEncoder.matches(signinRequest.getPassword(), _user.getPassword())) {
            return ResponseEntity.badRequest().body("Invalid email or password");
        }

        if (!_user.isEmailVerified()) {
            return ResponseEntity.badRequest().body("Please verify your email before logging in.");
        }

        String token = jwtUtil.generateToken(_user.getUsername());

        return ResponseEntity.ok(Map.of(
                "token", token,
                "user", Map.of(
                        "id", _user.getId(),
                        "username", _user.getUsername(),
                        "email", _user.getEmail(),
                        "roles", _user.getRoles().stream().map(Role::getName).toList()
                )
        ));
    }
    @GetMapping("/verify-email")
    public ResponseEntity<?> verifyEmail(@RequestParam("token") String token) {

        VerificationToken vt = verificationTokenRepo.findByToken(token)
                .orElseThrow(() -> new RuntimeException("Invalid verification token"));

        if (vt.getExpiresAt().isBefore(Instant.now())) {
            return ResponseEntity.badRequest().body("Verification token expired");
        }

        User user = vt.getUser();
        user.setEmailVerified(true);
        userRepo.save(user);

        verificationTokenRepo.delete(vt);

        return ResponseEntity.ok("Email verified successfully. You can now log in.");
    }
    @PostMapping("/resend-verification")
    public ResponseEntity<?> resendVerification(@RequestBody ResendVerificationRequest req) {

        if (req.getEmail() == null || req.getEmail().isBlank()) {
            return ResponseEntity.badRequest().body("Email is required");
        }

        User user = userRepo.findByEmail(req.getEmail())
                .orElseThrow(() -> new RuntimeException("No account found for that email"));

        if (user.isEmailVerified()) {
            return ResponseEntity.ok("Email is already verified. You can log in.");
        }

        // remove old token (so only one active token exists)
        verificationTokenRepo.deleteByUser(user);

        VerificationToken vt = new VerificationToken();
        vt.setToken(UUID.randomUUID().toString());
        vt.setUser(user);
        vt.setExpiresAt(Instant.now().plus(Duration.ofHours(24)));
        verificationTokenRepo.save(vt);

        String verifyLink = "http://localhost:5173/verify-email?token=" + vt.getToken();

        emailService.send(
                user.getEmail(),
                "Verify your AgriVerse account (new link)",
                "Here is your new verification link:\n" + verifyLink + "\n\nThis link expires in 24 hours."
        );

        return ResponseEntity.ok("Verification email resent. Please check your inbox.");
    }


}
