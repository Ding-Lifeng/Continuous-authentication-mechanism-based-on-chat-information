package com.example.server.mapper;

import com.example.server.entity.ChatRelation;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Mapper
@Repository
public interface ChatRelationMapper {
    List<ChatRelation> findChatRelationsByUid(Long uid);
}
