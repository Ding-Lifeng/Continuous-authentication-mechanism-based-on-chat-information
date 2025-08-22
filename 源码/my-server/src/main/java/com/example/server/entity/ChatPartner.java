package com.example.server.entity;

import lombok.Data;

@Data
public class ChatPartner {
    private Long id;
    private String name;
    private String lastMessage;
    private String lastMessageTime;
}
