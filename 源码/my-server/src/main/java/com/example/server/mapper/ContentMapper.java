package com.example.server.mapper;

import com.example.server.entity.ChatContent;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;
import java.util.List;

@Mapper
@Repository
public interface ContentMapper {
    void insertContent(ChatContent chatContent);

    void deleteContent(ChatContent chatContent);

    void updateContent(ChatContent chatContent);

    List<ChatContent> getContentByChatPartner(Long uid, Long chatPartnerId);

    List<ChatContent> getHistoryByUid(Long uid);

    ChatContent getLastContentByChatPartner(Long uid, Long chatPartnerId);

    void changeContentTag(Long uid, String content);

    void updateUserContent(Long uid);

    void deleteOutOfDateContent(Long uid);
}
