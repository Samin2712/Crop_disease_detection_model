package com.example.agriverse.repository;

import com.example.agriverse.model.ForumTopic;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ForumTopicRepository extends JpaRepository<ForumTopic, Long> {
    Optional<ForumTopic> findByNameIgnoreCase(String name);
    boolean existsByNameIgnoreCase(String name);
}
