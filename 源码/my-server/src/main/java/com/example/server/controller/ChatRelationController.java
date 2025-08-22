package com.example.server.controller;

import com.example.server.common.CommonResult;
import com.example.server.entity.ChatPartner;
import com.example.server.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.example.server.util.JwtTokenUtil;

import java.util.List;

@RestController
@RequestMapping("/chatRelation")
public class ChatRelationController {

    @Autowired
    private UserService userService;

    @Autowired
    private JwtTokenUtil jwtTokenUtil;

    @GetMapping("/getChatPartners")
    public CommonResult<?> getChatPartners(@RequestHeader("Authorization") String token) {
        try {
            String tokenValue = token.split(" ")[1];  // 获取 token
            long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(tokenValue));
            // 获取当前用户的聊天伙伴信息
            List<ChatPartner> chatPartners = userService.getChatPartnersByUid(uid);

            return CommonResult.success(chatPartners);  // 返回聊天伙伴列表
        } catch (Exception e) {
            return CommonResult.error(401,"获取聊天伙伴失败!");
        }
    }
}
