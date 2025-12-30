package com.example.agriverse.repository;

import com.example.agriverse.model.UserRequestMessage;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRequestMessageRepository extends JpaRepository<UserRequestMessage, Long> {
    Page<UserRequestMessage> findByRequestIdOrderByCreatedAtAsc(Long requestId, Pageable pageable);
}
