package com.example.agriverse.security;

import com.example.agriverse.service.CustomUserDetailsService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
@RequiredArgsConstructor
public class JwtFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;
    private final CustomUserDetailsService userDetailsService;

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        return request.getServletPath().startsWith("/auth/");
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain)
            throws ServletException, IOException {

        String authHeader = request.getHeader("Authorization");
        String token = null;
        String username = null;

        System.out.println("[JwtFilter] Path: " + request.getServletPath());
        System.out.println("[JwtFilter] Authorization header: " + authHeader);

        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            token = authHeader.substring(7);
            System.out.println("[JwtFilter] Token: " + token);
            try {
                username = jwtUtil.extractUsername(token);
                System.out.println("[JwtFilter] Username from token: " + username);
            } catch (Exception e) {
                System.out.println("[JwtFilter] Error extracting username: " + e.getMessage());
            }
        }

        if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {
            UserDetails userDetails = userDetailsService.loadUserByUsername(username);
            System.out.println("[JwtFilter] Loaded userDetails for: " + userDetails.getUsername());

            if (jwtUtil.validateToken(token)) {
                System.out.println("[JwtFilter] Token is valid. Setting authentication.");
                UsernamePasswordAuthenticationToken authToken = new UsernamePasswordAuthenticationToken(userDetails,
                        null, userDetails.getAuthorities());

                authToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                SecurityContextHolder.getContext().setAuthentication(authToken);
            } else {
                System.out.println("[JwtFilter] Token is invalid.");
            }
        } else {
            if (username == null) {
                System.out.println("[JwtFilter] Username is null. Token may be missing or invalid.");
            } else if (SecurityContextHolder.getContext().getAuthentication() != null) {
                System.out.println("[JwtFilter] Authentication already set.");
            }
        }
        filterChain.doFilter(request, response);
    }
}
