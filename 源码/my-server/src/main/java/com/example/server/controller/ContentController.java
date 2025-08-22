package com.example.server.controller;

import com.example.server.common.CommonResult;
import com.example.server.entity.ChatContent;
import com.example.server.util.JwtTokenUtil;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.concurrent.FutureCallback;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.nio.client.CloseableHttpAsyncClient;
import org.apache.http.impl.nio.client.HttpAsyncClients;
import org.apache.http.util.EntityUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.example.server.service.ContentService;
import com.example.server.service.AuthenticationService;
import org.apache.http.entity.ContentType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/chatContent")
public class ContentController {
    @Autowired
    private ContentService contentService;

    @Autowired
    private AuthenticationService authenticationService;

    @Autowired
    private JwtTokenUtil jwtTokenUtil;

    @PostMapping("/add")
    public CommonResult<?> addContent(@RequestHeader("Authorization") String token, @RequestBody ChatContent chatContent) throws IOException {
        long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(token.split(" ")[1]));
        chatContent.setUid(uid);

        if (authenticationService.isUserOnTheList(uid)) {
            return CommonResult.success("banned");
        }

        contentService.addContent(chatContent);

        // 提交认证模型进行认证
        String userHistory = contentService.getHistoryByUid(uid);
        String currentText = chatContent.getContent();

        // 创建 HttpAsyncClient 实例
        CloseableHttpAsyncClient httpClient = HttpAsyncClients.createDefault();
        httpClient.start();

        // 构建 POST 请求
        String predictUrl = "http://localhost:28081/predict/";
        HttpPost request = new HttpPost(predictUrl);
        request.addHeader("Content-Type", "application/json; charset=UTF-8");

        // 构建请求体
        String jsonPayload = String.format("{\"uid\": %d, \"user_history\": \"%s\", \"current_text\": \"%s\"}", uid, userHistory, currentText);
        StringEntity entity = new StringEntity(
                jsonPayload,
                ContentType.APPLICATION_JSON.withCharset(StandardCharsets.UTF_8)
        );
        request.setEntity(entity);

        // 使用 CompletableFuture 包装异步操作
        CompletableFuture<Void> responseFuture = new CompletableFuture<>();

        // 异步发送请求
        httpClient.execute(request, new FutureCallback<HttpResponse>() {
            @Override
            public void completed(HttpResponse httpResponse) {
                try {
                    if (httpResponse.getStatusLine().getStatusCode() == 200) {
                        // 成功响应
                        String responseBody = EntityUtils.toString(httpResponse.getEntity());
                        System.out.println("预测服务响应: " + responseBody);
                    } else {
                        // 失败响应
                        System.err.println("预测服务错误: " + httpResponse.getStatusLine());
                    }
                    responseFuture.complete(null);  // 完成 Future
                } catch (Exception e) {
                    responseFuture.completeExceptionally(e);  // 处理异常
                }
            }

            @Override
            public void failed(Exception e) {
                // 请求失败时
                System.err.println("HTTP请求失败: " + e.getMessage());
                responseFuture.completeExceptionally(e);
            }

            @Override
            public void cancelled() {
                // 请求被取消时
                System.err.println("HTTP请求取消");
                responseFuture.completeExceptionally(new RuntimeException("Request was cancelled"));
            }
        });

        // 等待请求完成
        responseFuture.join();

        // 关闭 HttpClient
        httpClient.close();

        String data = "Add Success";
        return CommonResult.success(data);
    }

    @GetMapping("/get")
    public CommonResult<?> getContent(
            @RequestHeader("Authorization") String token,
            @RequestParam Long chatPartnerId
    ) {
        // 从 Token 中解析当前用户 ID
        long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(token.split(" ")[1]));

        // 获取聊天记录
        List<ChatContent> data = contentService.getContentByChatPartner(uid, chatPartnerId);

        // 格式化返回数据
        List<Map<String, Object>> formattedData = data.stream().map(content -> {
            Map<String, Object> message = new HashMap<>();
            message.put("id", content.getContent_id());
            message.put("chatId", chatPartnerId);
            message.put("sender", content.getUid() == uid ? "me" : "partner"); // 根据发送者 ID 判定是当前用户还是对方
            message.put("text", content.getContent());
            message.put("time", content.getTime());
            return message;
        }).collect(Collectors.toList());

        return CommonResult.success(formattedData);
    }

    @PostMapping("/delete")
    public CommonResult<?> deleteContent(@RequestHeader("Authorization") String token, @RequestBody ChatContent chatContent) {
        long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(token.split(" ")[1]));
        chatContent.setUid(uid);
        contentService.deleteContent(chatContent);
        String data = "Delete Success";
        return CommonResult.success(data);
    }

    @PostMapping("/update")
    public CommonResult<?> updateContent(@RequestHeader("Authorization") String token, @RequestBody ChatContent chatContent) {
        long uid = Long.parseLong(jwtTokenUtil.getUidFromToken(token.split(" ")[1]));
        chatContent.setUid(uid);
        contentService.updateContent(chatContent);
        String data = "Update Success";
        return CommonResult.success(data);
    }

}
