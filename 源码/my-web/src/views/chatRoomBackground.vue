<template>
  <div class="chat-room-background">
    <!-- 用户头像 -->
    <div class="avatar-container">
      <el-avatar :src="userAvatar" class="user-avatar"></el-avatar>
    </div>

    <!-- 空白区域占位 -->
    <div class="empty-space">
      <span class="no-chat-message">欢迎使用Trusted Chat！</span>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      userAvatar: require('@/assets/Logo.jpeg'), // 当前用户头像
    };
  },
};
</script>

<style scoped lang="scss">
.chat-room-background {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f0f2f5; /* 与侧边栏颜色协调 */
  color: #333;
}

.avatar-container {
  margin-bottom: 20px;
}

.user-avatar {
  width: 80px;
  height: 80px;
  display: flex;
  justify-content: center;
  object-fit: cover;
}

.empty-space {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px; /* 空白区域高度 */
}

.no-chat-message {
  font-size: 18px;
  font-weight: bold;
}
</style>
