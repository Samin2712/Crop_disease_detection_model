package com.example.agriverse.model;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Table(name = "forum_comments")
@Getter @Setter
@NoArgsConstructor @AllArgsConstructor @Builder
public class ForumComment {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "forum_comment_seq_gen")
    @SequenceGenerator(name = "forum_comment_seq_gen", sequenceName = "forum_comment_seq", allocationSize = 1)
    private Long id;

    @ManyToOne(optional = false, fetch = FetchType.LAZY)
    @JoinColumn(name = "post_id", nullable = false)
    private ForumPost post;

    @ManyToOne(optional = false, fetch = FetchType.LAZY)
    @JoinColumn(name = "author_id", nullable = false)
    private User author;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @Column(nullable = false)
    private Instant createdAt;

    @PrePersist
    void onCreate() {
        createdAt = Instant.now();
    }
}
