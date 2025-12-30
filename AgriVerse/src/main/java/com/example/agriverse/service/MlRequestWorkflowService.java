package com.example.agriverse.service;

import com.example.agriverse.dto.UserRequestResponse;
import com.example.agriverse.dto.ml.MlPredictionResponse;
import com.example.agriverse.dto.ml.PredictAndCreateResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class MlRequestWorkflowService {

    private final MlPredictionService mlPredictionService;
    private final AiAdviceService aiAdviceService;
    private final UserRequestService userRequestService;

    public PredictAndCreateResponse predictAndCreate(
            String crop,
            MultipartFile image,
            String state,
            String district
    ) {
        MlPredictionResponse pred = mlPredictionService.predict(crop, image);

        if (pred == null || pred.error != null) {
            return PredictAndCreateResponse.builder()
                    .prediction(pred)
                    .advice(null)
                    .request(null)
                    .requestTopic(null)
                    .build();
        }

        String top1 = (pred.prediction != null) ? pred.prediction : "Unknown";
        String topic = crop + " • " + top1;

        // top 3 list
        List<MlPredictionResponse.TopK> topk = (pred.topk != null) ? pred.topk : List.of();
        String top3 = topk.stream()
                .limit(3)
                .map(t -> "- " + t.label + (t.score != null ? String.format(" (%.2f%%)", t.score * 100.0) : ""))
                .collect(Collectors.joining("\n"));

        // AI advice (best effort)
        String advice = aiAdviceService.getAdvice(crop, top1);

        StringBuilder desc = new StringBuilder();
        desc.append("Crop: ").append(crop).append("\n");
        desc.append("Most probable disease: ").append(top1).append("\n\n");
        desc.append("Top 3 guesses:\n").append(top3.isBlank() ? "- (no probabilities provided)" : top3).append("\n\n");
        desc.append("AI advice:\n").append(advice != null ? advice : "(AI advice not available)");

        // ✅ uses your existing create pipeline (saves image + OPEN status + first message)
        UserRequestResponse created = userRequestService.createRequestWithPhoto(
                topic,
                desc.toString(),
                state,
                district,
                image
        );

        return PredictAndCreateResponse.builder()
                .prediction(pred)
                .advice(advice)
                .requestTopic(topic)
                .request(created)
                .build();
    }
}
