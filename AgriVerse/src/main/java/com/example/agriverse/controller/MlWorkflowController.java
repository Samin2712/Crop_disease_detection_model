package com.example.agriverse.controller;

import com.example.agriverse.dto.ml.MlPredictionResponse;
import com.example.agriverse.service.MlPredictionService;
import com.example.agriverse.service.UserRequestService;
import org.springframework.web.bind.annotation.RequestBody;
import java.util.Map;

import com.example.agriverse.dto.ml.PredictAndCreateResponse;
import com.example.agriverse.service.MlRequestWorkflowService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import com.example.agriverse.service.AiAdviceService;
import com.example.agriverse.dto.UserRequestResponse;

@RestController
@RequestMapping("/api/ml")
@RequiredArgsConstructor
public class MlWorkflowController {
    private final UserRequestService userRequestService;
    private final MlRequestWorkflowService workflow;
    private final MlPredictionService mlPredictionService; // ✅ add this
    private final AiAdviceService aiAdviceService;

    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public MlPredictionResponse predict(
            @RequestPart("crop") String crop,
            @RequestPart("image") MultipartFile image
    ) {
        return mlPredictionService.predict(crop, image);
    }

    @PostMapping(value = "/predict-and-create", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public PredictAndCreateResponse predictAndCreate(
            @RequestPart("crop") String crop,
            @RequestPart("image") MultipartFile image,
            @RequestPart(value = "state", required = false) String state,
            @RequestPart(value = "district", required = false) String district
    ) {
        return workflow.predictAndCreate(crop, image, state, district);
    }

    @PostMapping("/advice")
    public Map<String, Object> advice(@RequestBody Map<String, String> body) {
        String cropName = body.get("crop_name");
        String diseaseName = body.get("disease_name");

        String answer = aiAdviceService.getAdvice(cropName, diseaseName);
        return Map.of("answer", answer);
    }

    @PostMapping(value = "/forward", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public UserRequestResponse forwardToGovtOfficer(
            @RequestPart("crop") String crop,
            @RequestPart("diseaseName") String diseaseName,
            @RequestPart("advice") String advice,
            @RequestPart("image") MultipartFile image,
            @RequestPart(value = "state", required = false) String state,
            @RequestPart(value = "district", required = false) String district
    ) {
        String category = crop + " • " + diseaseName;
        String description = diseaseName + "\n\n" + advice;

        return userRequestService.createRequestWithPhoto(category, description, state, district, image);
    }



}

