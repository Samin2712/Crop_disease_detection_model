package com.example.agriverse.service;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class AiAdviceService {

    private final WebClient genericWebClient;

    @Value("${ml.advice-webhook}")
    private String adviceWebhook;

    public String getAdvice(String cropName, String diseaseName) {
        try {
            // same payload shape as your temp JS
            Map<String, Object> payload = Map.of(
                    "crop_name", cropName,
                    "disease_name", diseaseName
            );

            // expects { answer: "..." } (fallback to raw)
            Map<?, ?> res = genericWebClient.post()
                    .uri(adviceWebhook)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(payload)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();

            if (res == null) return null;
            Object answer = res.get("answer");
            return (answer != null) ? String.valueOf(answer) : String.valueOf(res);
        } catch (Exception e) {
            return null; // don’t fail request creation if advice fails
        }
    }
}
