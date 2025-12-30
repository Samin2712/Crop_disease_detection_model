package com.example.agriverse.controller;

import com.example.agriverse.dto.*;
import com.example.agriverse.service.ForumService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/forum")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class ForumController {

    private final ForumService forumService;

    // TOPICS
    @GetMapping("/topics")
    public ResponseEntity<?> listTopics() {
        return ResponseEntity.ok(forumService.listTopics());
    }

    // Only admin can create new forum types like "cows", "rice"
    @PreAuthorize("hasRole('ADMIN')")
    @PostMapping("/topics")
    public ResponseEntity<?> createTopic(@RequestBody CreateTopicRequest req) {
        return ResponseEntity.ok(forumService.createTopic(req));
    }

    // POSTS
    @PreAuthorize("hasAnyRole('USER','GOVT_OFFICER')")
    @PostMapping("/posts")
    public ResponseEntity<?> createPost(@RequestBody CreatePostRequest req) {
        return ResponseEntity.ok(forumService.createPost(req));
    }

    @GetMapping("/topics/{topicId}/posts")
    public ResponseEntity<?> listPostsByTopic(
            @PathVariable Long topicId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String q
    ) {
        return ResponseEntity.ok(forumService.listPostsByTopic(topicId, page, size, q));
    }

    @GetMapping("/posts/{postId}")
    public ResponseEntity<?> getPost(@PathVariable Long postId) {
        return ResponseEntity.ok(forumService.getPost(postId));
    }

    // COMMENTS
    @PreAuthorize("hasAnyRole('USER','GOVT_OFFICER')")
    @PostMapping("/posts/{postId}/comments")
    public ResponseEntity<?> createComment(@PathVariable Long postId, @RequestBody CreateCommentRequest req) {
        return ResponseEntity.ok(forumService.createComment(postId, req));
    }

    @GetMapping("/posts/{postId}/comments")
    public ResponseEntity<?> listComments(
            @PathVariable Long postId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "30") int size
    ) {
        return ResponseEntity.ok(forumService.listComments(postId, page, size));
    }

    @PreAuthorize("hasAnyRole('USER','GOVT_OFFICER')")
    @GetMapping("/posts/mine")
    public ResponseEntity<?> myPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "5") int size
    ) {
        return ResponseEntity.ok(forumService.listMyRecentPosts(page, size));
    }

}
