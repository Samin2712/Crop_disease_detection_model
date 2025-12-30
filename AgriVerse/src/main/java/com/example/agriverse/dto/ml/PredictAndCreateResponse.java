package com.example.agriverse.dto.ml;

import com.example.agriverse.dto.UserRequestResponse;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class PredictAndCreateResponse {
    private MlPredictionResponse prediction;
    private String advice;             // n8n output text
    private String requestTopic;       // the final category/topic used
    private UserRequestResponse request; // created request
}
