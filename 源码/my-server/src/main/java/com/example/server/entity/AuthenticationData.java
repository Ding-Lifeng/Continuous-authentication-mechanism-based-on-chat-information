package com.example.server.entity;

import lombok.Data;

@Data
public class AuthenticationData {
    private long uid;  //  用户id
    private int result;  // 模型认证结果
    private String content;  // 文本内容
}
