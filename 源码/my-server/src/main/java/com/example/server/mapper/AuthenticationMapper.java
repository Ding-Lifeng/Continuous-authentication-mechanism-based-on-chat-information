package com.example.server.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.springframework.stereotype.Repository;
import org.apache.ibatis.annotations.Insert;

@Mapper
@Repository
public interface AuthenticationMapper {
    @Insert("INSERT INTO logout_list (uid) VALUES (#{uid})")
    void addUser(long uid);

    @Select("SELECT COUNT(*) FROM logout_list WHERE uid = #{uid}")
    int isUserOnTheList(long uid);

    @Insert("DELETE FROM logout_list WHERE uid = #{uid}")
    void deleteUser(long uid);
}
