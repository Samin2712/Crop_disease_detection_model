package com.example.agriverse.config;

import com.example.agriverse.model.Role;
import com.example.agriverse.repository.RoleRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class RoleConfig implements CommandLineRunner {

    private final RoleRepository roleRepository;

    public RoleConfig(RoleRepository roleRepository) {
        this.roleRepository = roleRepository;
    }

    @Override
    public void run(String... args) {
        // Add roles only if they don't exist
        if (!roleRepository.existsByName("ROLE_USER")) {
            roleRepository.save(new Role("ROLE_USER"));
        }
        if (!roleRepository.existsByName("ROLE_ADMIN")) {
            roleRepository.save(new Role("ROLE_ADMIN"));
        }
        if (!roleRepository.existsByName("ROLE_GOVT_OFFICER")) {
            roleRepository.save(new Role("ROLE_GOVT_OFFICER"));
        }
    }
}