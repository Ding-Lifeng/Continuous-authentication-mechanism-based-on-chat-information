package com.example.server.entity;

import lombok.Data;

@Data
public class ChatContent {
    private Long content_id;
    private Long uid;  // 用户id
    private Long chatPartner_id;  // 聊天对象id
    private String time;  // 文本上传时间
    private String content;  // 文本内容
    private Integer tag = 1;  // 标志位 1-文本由当前用户编写 0-文本的编写者待定
}
