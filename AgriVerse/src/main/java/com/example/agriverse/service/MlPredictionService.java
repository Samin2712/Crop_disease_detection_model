package com.example.agriverse.service;

import com.example.agriverse.dto.ml.MlPredictionResponse;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
public class MlPredictionService {

    private final WebClient mlWebClient;

    public MlPredictionService(WebClient mlWebClient) {
        this.mlWebClient = mlWebClient;
    }

    public MlPredictionResponse predict(String crop, MultipartFile image) {
        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("crop", crop);

            byte[] bytes = image.getBytes();
            builder.part("image", new ByteArrayResource(bytes) {
                        @Override public String getFilename() {
                            return (image.getOriginalFilename() != null) ? image.getOriginalFilename() : "leaf.jpg";
                        }
                    })
                    .contentType(MediaType.parseMediaType(
                            image.getContentType() != null ? image.getContentType() : MediaType.IMAGE_JPEG_VALUE
                    ));

            return mlWebClient.post()
                    .uri("/predict")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .bodyValue(builder.build())
                    .retrieve()
                    .bodyToMono(MlPredictionResponse.class)
                    .onErrorResume(ex -> {
                        MlPredictionResponse r = new MlPredictionResponse();
                        r.error = "ML service error: " + ex.getMessage();
                        return Mono.just(r);
                    })
                    .block();
        } catch (Exception e) {
            MlPredictionResponse r = new MlPredictionResponse();
            r.error = e.getMessage();
            return r;
        }
    }
}
