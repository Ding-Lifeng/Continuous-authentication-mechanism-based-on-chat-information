<template>
  <div>
    <div class="side-bar">
      <el-menu class="el-menu-vertical-demo" background-color="#545c64" text-color="#fff" active-text-color="#ffd04b">
        <!-- 用户头像和昵称 -->
        <div class="user-info">
          <el-avatar :src="userAvatar" class="user-avatar"></el-avatar>
          <span class="user-name">{{ userName }}</span>
        </div>

        <div class="chat-list">
          <!-- 聊天选择框 -->
          <div
              v-for="chat in chatList"
              :key="chat.id"
              class="chat-item"
              @click="selectChat(chat.id)"
          >
            <el-avatar class="chat-avatar">{{ chat.avatar }}</el-avatar>
            <div class="chat-info">
              <div class="chat-name">{{ chat.name }}</div>
              <div class="chat-last-message">{{ chat.lastMessage }}</div>
            </div>
          </div>
        </div>

        <!-- 退出按钮 -->
        <el-menu-item index="/logout" @click="logout">
          <i class="el-icon-setting"></i>
          <span slot="title">退出登陆</span>
        </el-menu-item>
      </el-menu>

      <div class="main-content">
        <router-view />
      </div>
    </div>
  </div>
</template>

<script>
import { getAccessToken, removeToken } from '@/utils/auth';
import { getUserInfo, getChatPartners } from "@/api/user";

export default {
  data() {
    return {
      isLoggedIn: false,
      userAvatar: require('@/assets/Logo.jpeg'),
      userName: '',
      chatList: [],
    };
  },
  created() {
    this.isLoggedIn = getAccessToken()
    this.fetchUserInfo();  // 获取用户信息
    this.fetchChatPartners();  // 获取聊天伙伴信息
  },
  methods: {
    generateAvatar(name) {
      return name.charAt(0).toUpperCase();
    },
    async fetchUserInfo() {
      try {
        const response = await getUserInfo();  // 获取用户信息
        if (response.data) {
          this.userName = response.data.name;  // 设置用户名
        }
      } catch (error) {
        console.error('Failed to fetch user info', error);
      }
    },
    async fetchChatPartners() {
      try {
        const response = await getChatPartners();
        if (response.data) {
          this.chatList = response.data.map(chat => ({
            ...chat,
            avatar: this.generateAvatar(chat.name),
          }));
        }
      } catch (error) {
        console.error('Failed to fetch chat partners', error);
      }
    },
    selectChat(chatPartnerId) {
      this.$router.push({ path: `/chat/${chatPartnerId}` });
    },
    logout() {
      this.$confirm('确定注销并退出系统吗？', '提示')
          .then(() => {
            removeToken(); // 清除 token
            this.$router.push({ path: '/login' }); // 重定向到登录页面
          })
          .catch(() => {});
    },
  },
};
</script>

<style lang="scss" scoped>
.side-bar {
  display: flex;
  height: 100vh;
}

.user-info {
  display: flex;
  align-items: center;
  padding: 20px;
  background-color: #39424e;
  color: #fff;
  border-bottom: 1px solid #2c3e50;
}

.user-avatar {
  margin-right: 10px;
  width: 80px;
  height: 80px;
  border-radius: 50%;
  border: 3px solid white;
  margin-bottom: 1rem;
  display: flex;
  justify-content: center;
  object-fit: cover;
}

.user-name {
  font-size: 16px;
  font-weight: bold;
}

.chat-list {
  margin: 20px;
}

.chat-item {
  display: flex;
  align-items: center;
  padding: 10px;
  cursor: pointer;
  transition: background-color 0.3s;

  &:hover {
    background-color: #2c3e50;
  }
}

.chat-avatar {
  margin-right: 10px;
}

.chat-info {
  display: flex;
  flex-direction: column;
}

.chat-name {
  font-size: 14px;
  font-weight: bold;
  color: #fff;
}

.chat-last-message {
  font-size: 12px;
  color: #ccc;
}

.main-content {
  flex: 1;
  padding: 20px;
}
</style>
