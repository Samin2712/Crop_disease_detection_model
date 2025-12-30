package com.example.agriverse.repository;

import com.example.agriverse.model.ForumComment;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ForumCommentRepository extends JpaRepository<ForumComment, Long> {
    Page<ForumComment> findByPostId(Long postId, Pageable pageable);
    long countByPostId(Long postId);
}
