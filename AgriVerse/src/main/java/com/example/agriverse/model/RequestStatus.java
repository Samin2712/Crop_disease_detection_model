package com.example.agriverse.model;

public enum RequestStatus {
    OPEN,          // created, waiting / not taken yet
    IN_PROGRESS,   // officer has taken it, chat active
    ARCHIVED       // solved & closed, chat locked
}
