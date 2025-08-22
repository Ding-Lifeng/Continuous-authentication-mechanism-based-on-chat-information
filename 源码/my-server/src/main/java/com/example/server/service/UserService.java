package com.example.server.service;

import com.example.server.entity.ChatContent;
import com.example.server.entity.ChatPartner;
import com.example.server.entity.ChatRelation;
import com.example.server.entity.User;
import com.example.server.mapper.ChatRelationMapper;
import com.example.server.mapper.UserMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import com.example.server.mapper.ContentMapper;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;


@Service
public class UserService {
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private ChatRelationMapper chatRelationMapper;
    @Autowired
    private ContentMapper contentMapper;


    public static String getMD5(String input, String salt) {
        try {
            input = input + salt; // 将密码和盐结合
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] messageDigest = md.digest(input.getBytes());
            StringBuilder hexString = new StringBuilder();
            for (byte b : messageDigest) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    @Transactional
    public void addUser(User user) {
        user.setPassword(getMD5(user.getPassword(), "行易xy"));
        userMapper.insertUser(user);
    }

    public void changeUserInfo(long uid, User user) {
        user.setPassword(getMD5(user.getPassword(), "行易xy"));
        userMapper.updateUserInfo(uid, user);
    }

    public boolean loginUser(String name, String password) {
        User user = userMapper.findUserByUsername(name);
        if (user != null) {
            // 验证密码
            String saltedPassword = getMD5(password, "行易xy");
            return saltedPassword.equals(user.getPassword());
        }
        return false;
    }

    public long findUserUid(String username){
        User user = userMapper.findUserByUsername(username);
        return user.getId();
    }

    public User getUserInfo(long uid){
        User existedUser = userMapper.findUserByUid(uid);
        // 清空密码-防止重复加密
        existedUser.setPassword(null);
        return existedUser;
    }

    public List<ChatPartner> getChatPartnersByUid(long uid) {
        List<ChatRelation> chatRelations = chatRelationMapper.findChatRelationsByUid(uid);

        List<ChatPartner> chatPartners = new ArrayList<>();
        for (ChatRelation relation : chatRelations) {
            long partnerId = relation.getChatPartner();
            String partnerName = userMapper.findUserByUid(partnerId).getName();

            // 使用 ContentMapper 获取最后一条消息
            ChatContent lastMessage = contentMapper.getLastContentByChatPartner(uid, partnerId);

            ChatPartner partner = new ChatPartner();
            partner.setId(partnerId);
            partner.setName(partnerName);
            partner.setLastMessage(lastMessage != null ? lastMessage.getContent() : "No messages");
            partner.setLastMessageTime(relation.getTime());

            chatPartners.add(partner);
        }

        // 按照最后聊天时间降序排列
        chatPartners.sort(Comparator.comparing(ChatPartner::getLastMessageTime).reversed());

        return chatPartners;
    }
}
