package com.example.agriverse.spec;

import com.example.agriverse.model.ForumPost;
import org.springframework.data.jpa.domain.Specification;

import jakarta.persistence.criteria.Expression;

public class ForumPostSpecs {

    public static Specification<ForumPost> byTopicId(Long topicId) {
        return (root, query, cb) -> cb.equal(root.get("topic").get("id"), topicId);
    }

    /**
     * Matches ALL words (any order) against title + content combined.
     * Example q="cow fever not eating"
     * -> title+content must contain cow AND fever AND not AND eating
     */
    public static Specification<ForumPost> containsAllWordsInTitleAndContent(String q) {
        if (q == null) q = "";
        final String trimmed = q.trim();
        if (trimmed.isEmpty()) {
            return (root, query, cb) -> cb.conjunction();
        }

        final String[] words = trimmed.toLowerCase().split("\\s+");

        return (root, query, cb) -> {
            Expression<String> haystack = cb.lower(
                    cb.concat(
                            cb.concat(root.get("title"), " "),
                            root.get("content")
                    )
            );

            // AND all words
            var predicate = cb.conjunction();
            for (String w : words) {
                if (w.isBlank()) continue;
                predicate = cb.and(predicate, cb.like(haystack, "%" + w + "%"));
            }
            return predicate;
        };
    }
}
