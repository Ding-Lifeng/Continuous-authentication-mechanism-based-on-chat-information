package com.example.server.mapper;

import com.example.server.entity.User;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;

@Mapper
@Repository
public interface UserMapper {
    void insertUser(User User);

    User findUserByUsername(String username);

    User findUserByUid(long uid);

    void updateUserInfo(long uid, User user);
}
