package com.example.server.controller;

import com.example.server.common.CommonResult;

import com.example.server.entity.User;
import com.example.server.service.AuthenticationService;
import com.example.server.service.ContentService;
import com.example.server.service.UserService;
import com.example.server.util.JwtTokenUtil;
import com.example.server.util.TokenResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/user")
public class UserController {
    @Autowired
    private UserService userService;

    @Autowired
    private JwtTokenUtil jwtTokenUtil;

    @Autowired
    private AuthenticationService authenticationService;

    @Autowired
    private ContentService contentService;

    @PostMapping("/add")
    public CommonResult<?> addUser(@RequestBody User user) {
        userService.addUser(user);
        String data = "用户添加成功!";
        return CommonResult.success(data);
    }

    @PostMapping("/change")
    public CommonResult<?> changeUserInfo(@RequestHeader("Authorization") String token, @RequestBody User user) {
        long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(token.split(" ")[1]));
        userService.changeUserInfo(uid, user);
        String data = "用户修改成功!";
        return CommonResult.success(data);
    }

    @GetMapping("/getInfo")
    public CommonResult<?> getUserInfo(@RequestHeader("Authorization") String token) {
        long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(token.split(" ")[1]));
        User data= userService.getUserInfo(uid);
        return CommonResult.success(data);
    }

    @PostMapping("/login")
    public CommonResult<?> login(@RequestBody User loginUser) {
        boolean success = userService.loginUser(loginUser.getName(), loginUser.getPassword());
        long uid;

        if (success) {
            uid = userService.findUserUid(loginUser.getName());
            // 生成访问令牌和刷新令牌
            String accessToken = jwtTokenUtil.generateAccessToken(Long.toString(uid));
            String refreshToken = jwtTokenUtil.generateRefreshToken(Long.toString(uid));
            TokenResponse token_resp = new TokenResponse(accessToken,refreshToken);

            // 成功登陆后解除用户封禁
            if (authenticationService.isUserOnTheList(uid)) {
                authenticationService.deleteUser(uid);
            }

            // 根据聊天信息更新用户文本特征
            contentService.updateUserContent(uid);

            return CommonResult.success(token_resp);
        } else {
            return CommonResult.error(401, "用户名或密码错误!");
        }
    }

    @GetMapping("/get/{chatPartnerId}")
    public CommonResult<?> getChatPartnerInfo(@PathVariable Long chatPartnerId) {
        String userName = userService.getUserInfo(chatPartnerId).getName();
        if (userName != null) {
            return  CommonResult.success(userName);
        } else {
            return CommonResult.error(404, "未找到用户!");
        }
    }
}
