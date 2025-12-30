package com.example.agriverse.model;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "forum_topics")
@Getter @Setter
@NoArgsConstructor @AllArgsConstructor @Builder
public class ForumTopic {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "forum_topic_seq_gen")
    @SequenceGenerator(name = "forum_topic_seq_gen", sequenceName = "forum_topic_seq", allocationSize = 1)
    private Long id;

    @Column(nullable = false, unique = true, length = 80)
    private String name; // e.g. "cows", "rice"

    @Column(length = 500)
    private String description;
}
