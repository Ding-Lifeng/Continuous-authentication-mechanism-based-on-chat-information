package com.example.server.controller;

import com.example.server.common.CommonResult;
import com.example.server.entity.AuthenticationData;
import com.example.server.service.ContentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.example.server.service.AuthenticationService;
import com.example.server.util.JwtTokenUtil;

@RestController
@RequestMapping("/Authentication")
public class AuthenticationController {
    @Autowired
    private AuthenticationService authenticationService;

    @Autowired
    private ContentService contentService;

    @Autowired
    private JwtTokenUtil jwtTokenUtil;

    @PostMapping("/add")
    public CommonResult<?> addAuthenticationInfo(@RequestBody AuthenticationData authenticationData) {
        long uid = authenticationData.getUid();
        int result = authenticationData.getResult();
        String content = authenticationData.getContent();

        if(result == 0){
            authenticationService.addUser(uid);
            contentService.changeContentTag(uid, content);
            return CommonResult.success("用户进入登出列表");
        }
        else {
            return CommonResult.success("用户状态正常");
        }
    }

    @GetMapping("/checkStatus")
    public CommonResult<?> checkUserStatus(@RequestHeader("Authorization") String token) {
        long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(token.split(" ")[1]));
        boolean isBanned = authenticationService.isUserOnTheList(uid);
        return CommonResult.success(isBanned ? "banned" : "normal");
    }
}
