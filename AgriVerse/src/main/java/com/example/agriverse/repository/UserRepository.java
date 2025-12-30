package com.example.agriverse.repository;

import com.example.agriverse.model.User;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);

    Boolean existsByEmail(String email);

    Optional<User> findByUsername(String username);

    Boolean existsByUsername(String username);

    Boolean existsByIdentificationNumber(String identificationNumber);

    Optional<User> findFirstByRoles_NameOrderByIdAsc(String roleName);

    List<User> findByRoles_Name(String roleName);

}
