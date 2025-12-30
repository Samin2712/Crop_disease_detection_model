package com.example.agriverse.service;

import com.example.agriverse.dto.*;
import com.example.agriverse.model.*;
import com.example.agriverse.repository.*;
import com.example.agriverse.spec.ForumPostSpecs;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.*;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class ForumService {

    private final ForumTopicRepository topicRepo;
    private final ForumPostRepository postRepo;
    private final ForumCommentRepository commentRepo;
    private final UserRepository userRepo;

    private User currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getName() == null) throw new RuntimeException("Unauthorized");
        return userRepo.findByUsername(auth.getName()).orElseThrow(() -> new RuntimeException("User not found"));
    }

    // TOPICS
    public TopicResponse createTopic(CreateTopicRequest req) {
        String name = req.getName() == null ? null : req.getName().trim();
        if (name == null || name.isBlank()) throw new RuntimeException("Topic name is required");
        if (topicRepo.existsByNameIgnoreCase(name)) throw new RuntimeException("Topic already exists");

        ForumTopic saved = topicRepo.save(ForumTopic.builder()
                .name(name)
                .description(req.getDescription())
                .build());

        return TopicResponse.builder()
                .id(saved.getId())
                .name(saved.getName())
                .description(saved.getDescription())
                .postCount(0)
                .build();
    }

    public java.util.List<TopicResponse> listTopics() {
        return topicRepo.findAll(Sort.by(Sort.Direction.ASC, "name"))
                .stream()
                .map(t -> TopicResponse.builder()
                        .id(t.getId())
                        .name(t.getName())
                        .description(t.getDescription())
                        .postCount(postRepo.countByTopicId(t.getId()))
                        .build())
                .toList();
    }

    // POSTS
    public PostResponse createPost(CreatePostRequest req) {
        if (req.getTopicId() == null) throw new RuntimeException("topicId is required");
        if (req.getTitle() == null || req.getTitle().trim().isBlank()) throw new RuntimeException("title is required");
        if (req.getContent() == null || req.getContent().trim().isBlank()) throw new RuntimeException("content is required");

        ForumTopic topic = topicRepo.findById(req.getTopicId())
                .orElseThrow(() -> new RuntimeException("Topic not found"));

        User author = currentUser();

        ForumPost saved = postRepo.save(ForumPost.builder()
                .topic(topic)
                .author(author)
                .title(req.getTitle().trim())
                .content(req.getContent().trim())
                .build());

        return toPostResponse(saved);
    }

    public Page<PostResponse> listPostsByTopic(Long topicId, int page, int size, String q) {
        Specification<ForumPost> spec =
                ForumPostSpecs.byTopicId(topicId)
                        .and(ForumPostSpecs.containsAllWordsInTitleAndContent(q));

        Page<ForumPost> posts = postRepo.findAll(
                spec,
                PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "createdAt"))
        );

        return posts.map(this::toPostResponse);
    }

    public PostResponse getPost(Long postId) {
        ForumPost post = postRepo.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found"));
        return toPostResponse(post);
    }

    // COMMENTS
    public CommentResponse createComment(Long postId, CreateCommentRequest req) {
        if (req.getContent() == null || req.getContent().trim().isBlank()) {
            throw new RuntimeException("content is required");
        }
        ForumPost post = postRepo.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found"));

        User author = currentUser();
        ForumComment saved = commentRepo.save(ForumComment.builder()
                .post(post)
                .author(author)
                .content(req.getContent().trim())
                .build());

        return CommentResponse.builder()
                .id(saved.getId())
                .postId(postId)
                .authorUsername(author.getUsername())
                .content(saved.getContent())
                .createdAt(saved.getCreatedAt())
                .build();
    }

    public Page<CommentResponse> listComments(Long postId, int page, int size) {
        return commentRepo.findByPostId(
                postId,
                PageRequest.of(page, size, Sort.by(Sort.Direction.ASC, "createdAt"))
        ).map(c -> CommentResponse.builder()
                .id(c.getId())
                .postId(postId)
                .authorUsername(c.getAuthor().getUsername())
                .content(c.getContent())
                .createdAt(c.getCreatedAt())
                .build());
    }

    private PostResponse toPostResponse(ForumPost p) {
        return PostResponse.builder()
                .id(p.getId())
                .topicId(p.getTopic().getId())
                .topicName(p.getTopic().getName())
                .title(p.getTitle())
                .content(p.getContent())
                .authorUsername(p.getAuthor().getUsername())
                .createdAt(p.getCreatedAt())
                .updatedAt(p.getUpdatedAt())
                .commentCount(commentRepo.countByPostId(p.getId()))
                .build();
    }
    public Page<PostResponse> listMyRecentPosts(int page, int size) {
        User user = currentUser();

        Page<ForumPost> posts = postRepo.findByAuthorUsername(
                user.getUsername(),
                PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "createdAt"))
        );

        return posts.map(this::toPostResponse);
    }

}
