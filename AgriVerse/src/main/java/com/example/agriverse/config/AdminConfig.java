package com.example.agriverse.config;

import com.example.agriverse.model.Role;
import com.example.agriverse.model.User;
import com.example.agriverse.repository.RoleRepository;
import com.example.agriverse.repository.UserRepository;
import lombok.AllArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.util.Set;

@Component
@AllArgsConstructor
public class AdminConfig implements CommandLineRunner {
    
    private final UserRepository userRepository;
    private final RoleRepository roleRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    public void run(String... args) {
        Role userRole = roleRepository.findByName("ROLE_USER")
                .orElseGet(() -> roleRepository.save(new Role("ROLE_USER")));

        Role adminRole = roleRepository.findByName("ROLE_ADMIN")
                .orElseGet(() -> roleRepository.save(new Role("ROLE_ADMIN")));
        Role govtOfficerRole = roleRepository.findByName("ROLE_GOVT_OFFICER")
                .orElseGet(() -> roleRepository.save(new Role("ROLE_GOVT_OFFICER")));


        if (!userRepository.existsByUsername("admin")) {
            User user = new User();
            user.setUsername("admin");
            user.setEmail("admin@example.com"); // better to use a real email format
            user.setPassword(passwordEncoder.encode("admin")); // never store plain text
            user.setRoles(Set.of(userRole, adminRole));
            user.setEmailVerified(true);

            userRepository.save(user);
            System.out.println("✅ Admin user created: admin / admin");
        } else {
            System.out.println("ℹ️ Admin user already exists");
        }
    }

}
