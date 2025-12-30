package com.example.agriverse.repository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import com.example.agriverse.model.ForumPost;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;

public interface ForumPostRepository extends JpaRepository<ForumPost, Long>, JpaSpecificationExecutor<ForumPost> {
    long countByTopicId(Long topicId);

    Page<ForumPost> findByAuthorUsername(
            String username,
            Pageable pageable
    );

}


