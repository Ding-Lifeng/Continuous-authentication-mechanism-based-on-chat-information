package com.example.server.service;

import com.example.server.mapper.AuthenticationMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class AuthenticationService {
    @Autowired
    private AuthenticationMapper authenticationMapper;

    @Transactional
    public void addUser(Long uid) {
        authenticationMapper.addUser(uid);
    }

    public boolean isUserOnTheList(Long uid) {
        int count = authenticationMapper.isUserOnTheList(uid);
        return count == 1;
    }

    public void deleteUser(Long uid) {
        authenticationMapper.deleteUser(uid);
    }
}
