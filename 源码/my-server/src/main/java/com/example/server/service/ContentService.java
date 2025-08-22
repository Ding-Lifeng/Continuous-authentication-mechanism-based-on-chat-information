package com.example.server.service;

import com.example.server.entity.ChatContent;
import com.example.server.mapper.ContentMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class ContentService {
    @Autowired
    private ContentMapper contentMapper;

    @Transactional
    public void addContent(ChatContent chatContent) {
        contentMapper.insertContent(chatContent);
    }

    public void deleteContent(ChatContent chatContent) {
        contentMapper.deleteContent(chatContent);
    }

    public void updateContent(ChatContent chatContent) {
        contentMapper.updateContent(chatContent);
    }

    public String getHistoryByUid(Long uid) {
        // 从数据库中查询所有内容
        List<ChatContent> contents = contentMapper.getHistoryByUid(uid);

        StringBuilder historyContents = new StringBuilder();

        for (ChatContent content : contents) {
            historyContents.append(content.getContent()).append(" ");
        }

        return historyContents.toString().trim();
    }

    public List<ChatContent> getContentByChatPartner(Long uid, Long chatPartnerId) {
        return contentMapper.getContentByChatPartner(uid, chatPartnerId);
    }

    public void changeContentTag(Long uid, String content) {
        // 修改指定用户指定数据的最新一条记录的标志位
        contentMapper.changeContentTag(uid, content);
    }

    @Transactional
    public void updateUserContent(Long uid) {
        // 修改指定用户指定数据的最新一条记录的标志位
        contentMapper.updateUserContent(uid);
        contentMapper.deleteOutOfDateContent(uid);
    }
}
