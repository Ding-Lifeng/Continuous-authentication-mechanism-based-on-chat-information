package com.example.server.entity;

import lombok.Data;

@Data
public class ChatRelation {
    private Long uid;
    private Long chatPartner;
    private String time;
}
