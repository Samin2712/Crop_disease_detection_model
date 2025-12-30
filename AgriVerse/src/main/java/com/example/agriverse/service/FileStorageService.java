package com.example.agriverse.service;

import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.util.UUID;

@Service
public class FileStorageService {

    private final Path uploadRoot = Paths.get("uploads");

    public String saveImage(MultipartFile file) {
        if (file == null || file.isEmpty()) return null;

        try {
            Files.createDirectories(uploadRoot);

            String original = StringUtils.cleanPath(file.getOriginalFilename() == null ? "image" : file.getOriginalFilename());
            String ext = "";

            int dot = original.lastIndexOf('.');
            if (dot >= 0) ext = original.substring(dot);

            String filename = UUID.randomUUID() + ext;
            Path target = uploadRoot.resolve(filename).normalize();

            // basic safety: prevent path traversal
            if (!target.startsWith(uploadRoot)) throw new RuntimeException("Invalid file path");

            Files.copy(file.getInputStream(), target, StandardCopyOption.REPLACE_EXISTING);

            // URL your frontend can load
            return "/api/files/" + filename;

        } catch (IOException e) {
            throw new RuntimeException("Failed to store file", e);
        }
    }

    public Path getFilePath(String filename) {
        return uploadRoot.resolve(filename).normalize();
    }
}
