package com.example.agriverse.repository;

import com.example.agriverse.model.RequestStatus;
import com.example.agriverse.model.UserRequest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRequestRepository extends JpaRepository<UserRequest, Long> {

    Page<UserRequest> findByCreatedByUsername(String username, Pageable pageable);

    Page<UserRequest> findByAssignedOfficerUsername(String username, Pageable pageable);

    // Officer queue: requests not taken yet
    Page<UserRequest> findByStatusAndAssignedOfficerIsNull(RequestStatus status, Pageable pageable);

    Page<UserRequest> findByAssignedOfficerUsernameAndStatusNot(
            String username, RequestStatus status, Pageable pageable
    );


    // Archived for user
    Page<UserRequest> findByCreatedByUsernameAndStatus(String username, RequestStatus status, Pageable pageable);

    // Archived for officer
    Page<UserRequest> findByAssignedOfficerUsernameAndStatus(String username, RequestStatus status, Pageable pageable);
}
