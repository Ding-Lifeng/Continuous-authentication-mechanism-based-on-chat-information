package com.example.server.entity;

import lombok.Data;

@Data
public class User {
    private Long id;
    private String name;
    private String password;
    private String real_name;
    private String gender;
    private Integer phoneNum;
}

