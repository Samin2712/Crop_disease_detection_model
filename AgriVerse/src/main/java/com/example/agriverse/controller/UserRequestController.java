package com.example.agriverse.controller;

import com.example.agriverse.dto.ForwardUserRequestRequest;
import com.example.agriverse.dto.SendUserRequestMessageRequest;
import com.example.agriverse.service.UserRequestService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/requests")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class UserRequestController {

    private final UserRequestService requestService;

    // ✅ Create request with photo + description
    @PreAuthorize("hasAnyRole('USER','ADMIN')")
    @PostMapping(consumes = {"multipart/form-data"})
    public ResponseEntity<?> create(
            @RequestPart("category") String category,
            @RequestPart("description") String description,
            @RequestPart(value = "state", required = false) String state,
            @RequestPart(value = "district", required = false) String district,
            @RequestPart(value = "image", required = false) MultipartFile image
    ) {
        return ResponseEntity.ok(
                requestService.createRequestWithPhoto(category, description, state, district, image)
        );
    }

    // user's own requests
    @PreAuthorize("hasAnyRole('USER','ADMIN','GOVT_OFFICER')")
    @GetMapping("/mine")
    public ResponseEntity<?> mine(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size
    ) {
        return ResponseEntity.ok(requestService.myRequests(page, size));
    }

    // officer queue (OPEN not taken)
    @PreAuthorize("hasAnyRole('GOVT_OFFICER','ADMIN')")
    @GetMapping("/officer/queue")
    public ResponseEntity<?> officerQueue(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size
    ) {
        return ResponseEntity.ok(requestService.officerQueue(page, size));
    }

    // officer takes a request -> chat active
    @PreAuthorize("hasAnyRole('GOVT_OFFICER','ADMIN')")
    @PostMapping("/{id}/take")
    public ResponseEntity<?> take(@PathVariable Long id) {
        return ResponseEntity.ok(requestService.takeRequest(id));
    }

    // chat: list messages
    @PreAuthorize("hasAnyRole('USER','GOVT_OFFICER','ADMIN')")
    @GetMapping("/{id}/messages")
    public ResponseEntity<?> messages(
            @PathVariable Long id,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "50") int size
    ) {
        return ResponseEntity.ok(requestService.getMessages(id, page, size));
    }

    // chat: send message
    @PreAuthorize("hasAnyRole('USER','GOVT_OFFICER','ADMIN')")
    @PostMapping("/{id}/messages")
    public ResponseEntity<?> sendMessage(
            @PathVariable Long id,
            @RequestBody SendUserRequestMessageRequest body
    ) {
        return ResponseEntity.ok(requestService.sendMessage(id, body));
    }

    // solved -> archive (locks chat)
    @PreAuthorize("hasAnyRole('GOVT_OFFICER','ADMIN')")
    @PostMapping("/{id}/archive")
    public ResponseEntity<?> archive(@PathVariable Long id) {
        return ResponseEntity.ok(requestService.archiveRequest(id));
    }

    @PreAuthorize("hasAnyRole('USER','ADMIN')")
    @GetMapping("/mine/archived")
    public ResponseEntity<?> myArchived(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size
    ) {
        return ResponseEntity.ok(requestService.myArchived(page, size));
    }

    @PreAuthorize("hasAnyRole('GOVT_OFFICER','ADMIN')")
    @GetMapping("/officer/archived")
    public ResponseEntity<?> officerArchived(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size
    ) {
        return ResponseEntity.ok(requestService.officerArchived(page, size));
    }

    @PreAuthorize("hasAnyRole('USER','GOVT_OFFICER','ADMIN')")
    @GetMapping("/{id}")
    public ResponseEntity<?> getOne(@PathVariable Long id) {
        return ResponseEntity.ok(requestService.getRequest(id));
    }

    @PreAuthorize("hasAnyRole('GOVT_OFFICER','ADMIN')")
    @PostMapping("/{id}/forward")
    public ResponseEntity<?> forward(
            @PathVariable Long id,
            @RequestBody ForwardUserRequestRequest body
    ) {
        return ResponseEntity.ok(requestService.forwardRequest(id, body.getToOfficerUsername()));
    }


    @PreAuthorize("hasAnyRole('GOVT_OFFICER','ADMIN')")
    @GetMapping("/officer/assigned")
    public ResponseEntity<?> officerAssigned(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size
    ) {
        return ResponseEntity.ok(requestService.officerAssigned(page, size));
    }

}
