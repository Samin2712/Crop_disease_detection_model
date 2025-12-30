package com.example.agriverse.dto.ml;

import java.util.List;

public class MlPredictionResponse {
    public String crop;
    public String model;
    public String prediction;
    public Double confidence;
    public List<TopK> topk;
    public LeafGate leaf_gate;
    public String error;
    public Object details;

    public static class TopK {
        public String label;
        public Double score;
    }

    public static class LeafGate {
        public Object opencv_info;
        public Double ml_confidence;
    }
}
