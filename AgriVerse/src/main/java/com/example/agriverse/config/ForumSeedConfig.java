package com.example.agriverse.config;

import com.example.agriverse.model.ForumTopic;
import com.example.agriverse.repository.ForumTopicRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
public class ForumSeedConfig implements CommandLineRunner {

    private final ForumTopicRepository topicRepo;

    @Override
    public void run(String... args) {
        List<String> defaults = List.of("cows", "rice", "fish", "vegetables", "poultry");

        for (String name : defaults) {
            if (!topicRepo.existsByNameIgnoreCase(name)) {
                topicRepo.save(ForumTopic.builder()
                        .name(name)
                        .description("Forum for " + name)
                        .build());
            }
        }
    }
}
