package com.example.agriverse.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientsConfig {

    @Bean
    public WebClient mlWebClient(@Value("${ml.base-url}") String baseUrl) {
        return WebClient.builder().baseUrl(baseUrl).build();
    }

    @Bean
    public WebClient genericWebClient() {
        return WebClient.builder().build();
    }
}
