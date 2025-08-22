<template>
  <div class="chat-room">
    <!-- 聊天头部区域 -->
    <div class="header">
      <el-avatar class="chat-partner-avatar">{{ chatPartner.avatar }}</el-avatar>
      <span class="chat-partner-name">{{ chatPartner.name }}</span>
    </div>

    <!-- 聊天记录区域 -->
    <div class="messages-container" ref="messagesContainer">
      <div
          v-for="message in messages"
          :key="message.id"
          class="message"
          :class="{ 'sent': message.sender === 'me', 'received': message.sender !== 'me' }"
      >
        <el-avatar
            v-if="message.sender === 'me'"
            :src="userAvatar"
            class="message-avatar"
        />
        <span v-else class="chat-partner-avatar-message">
          {{ chatPartner.avatar }}
        </span>
        <div class="message-content">{{ message.text }}</div>
      </div>
    </div>

    <!-- 输入区域 -->
    <div class="input-container">
      <!-- 表情按钮 -->
      <el-button
          icon="el-icon-star-off"
          class="emoji-button"
          @click="toggleEmojiPicker"
      ></el-button>

      <!-- 输入框 -->
      <el-input
          v-model="newMessage"
          placeholder="请输入消息..."
          @keyup.enter="sendMessage"
          class="input-box"
          type="textarea"
          :rows="4"
      ></el-input>

      <!-- 发送按钮 -->
      <el-button @click="sendMessage" class="send-button">
        <span>发送</span>
      </el-button>
    </div>

    <!-- 表情选择区域 -->
    <div v-if="showEmojiPicker" class="emoji-picker">
      <VEmojiPicker @select="addEmoji" />
    </div>
  </div>
</template>

<script>
import { sendChatContent, getChatContent } from '@/api/chatContent.js'
import { getChatPartnerInfo } from '@/api/chatPartner.js'
import { removeToken } from "@/utils/auth";

export default {
  data() {
    return {
      userAvatar: require("@/assets/Logo.jpeg"), // 当前用户头像
      chatPartner: {},
      messages: [],
      newMessage: "",
      showEmojiPicker: false, // 控制表情选择器显示
      chatPartnerId: this.$route.params.chatPartnerId,
    };
  },
  watch: {
    '$route'(to, from) {
      if (to.params.chatPartnerId !== from.params.chatPartnerId) {
        this.chatPartnerId = to.params.chatPartnerId;  // 更新 chatPartnerId
        this.loadChatData(this.chatPartnerId);  // 获取聊天数据
      }
    },
  },
  methods: {
    // 聊天界面初始化
    async loadChatData(chatPartnerId) {
      // 获取聊天对象信息
      const chatResponse = await getChatPartnerInfo(chatPartnerId);
      console.log(chatResponse);
      const partnerName = chatResponse.data;
      this.chatPartner = {
        name: partnerName,
        avatar: partnerName.charAt(0).toUpperCase(),
      };

      // 获取两人之间的聊天记录
      this.messages = await this.fetchChatMessages(chatPartnerId);
    },
    async fetchChatMessages(chatPartnerId) {
      try {
        const response = await getChatContent(chatPartnerId); // 调用API获取两人聊天记录
        return response.data || [];
      } catch (error) {
        console.error("获取聊天记录失败", error);
        return [];
      }
    },

    // 发送聊天信息
    async sendMessage() {
      if (this.newMessage.trim()) {
        try {
          const response = await sendChatContent(this.chatPartnerId, this.newMessage.trim());

          // 检查响应是否为封禁状态
          if (response.data === "banned") {
            this.handleForceLogout();
            return; // 终止后续流程
          }

          // 正常处理消息
          this.messages.push({
            text: this.newMessage.trim(),
            id: response.data.id,
            sender: "me"
          });

          this.newMessage = "";
          this.$nextTick(() => {
            const container = this.$refs.messagesContainer;
            container.scrollTop = container.scrollHeight;
          });

        } catch (error) {
          console.error("发送消息失败", error);
        }
      }
    },
    handleForceLogout() {
      this.$confirm('您的账号已被强制登出', '提示', {
        confirmButtonText: '确定',
        showCancelButton: false,
        type: 'warning'
      }).then(() => {
        removeToken();
        this.$router.push('/login');
      });
    },
    toggleEmojiPicker() {  // Emoji表情
      this.showEmojiPicker = !this.showEmojiPicker; // 切换表情选择器显示状态
    },
    addEmoji(emoji) {
      // 直接使用 emoji.data 添加到消息
      this.newMessage += emoji.data;
    },
  },
  mounted() {
    // 初始加载聊天数据
    this.chatPartnerId = this.$route.params.chatPartnerId;
    this.loadChatData(this.chatPartnerId);

    // 初始加载后滚动到最新消息
    this.$nextTick(() => {
      const container = this.$refs.messagesContainer;
      container.scrollTop = container.scrollHeight;
    });
  },
};
</script>

<style scoped lang="scss">
.chat-room {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f0f2f5;
  border: 1px solid #dcdfe6;  /* 浅灰色边框 */
}

.header {
  display: flex;
  align-items: center;
  background-color: #39424e;
  padding: 15px;
  color: white;
  border-bottom: 1px solid #2c3e50;
}

.chat-partner-avatar {
  margin-right: 10px;
  width: 40px;
  height: 40px;
}

.chat-partner-name {
  font-size: 18px;
  font-weight: bold;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background-color: #fff;
}

.message {
  display: flex;
  align-items: flex-start;
  margin-bottom: 15px;
}

.sent .message-content {
  background-color: #d1f1d1;
  text-align: right;
}

.received .message-content {
  background-color: #f0f0f0;
  text-align: left;
}

.message-avatar {
  width: 35px;
  height: 35px;
  margin-right: 10px;
  display: flex;
  justify-content: center;
  object-fit: cover;
}

.message-content {
  max-width: 60%;
  padding: 10px;
  border-radius: 5px;
  font-size: 14px;
}

.input-container {
  display: flex;
  flex-direction: column;
  padding: 10px;
  background-color: #fff;
  border-top: 1px solid #ddd;
  position: relative;
}

.emoji-button {
  position: absolute;
  top: 125px;
  left: 10px;
  background-color: #409eff;
  color: white;
  border-radius: 50%;
  padding: 10px;
  font-size: 18px;
}

.input-box {
  flex: 1;
  margin-right: 10px;
  border-radius: 20px;
  font-size: 16px;
  resize: none;
}

.send-button {
  margin-top: 10px;
  background-color: #409eff;
  color: white;
  border-radius: 20px;
  padding: 10px 15px;
  align-self: flex-end; /* Align send button to the right */
}

.send-button span {
  text-align: center;
  display: inline-block;
}

.emoji-picker {
  position: absolute;
  bottom: 0;
  left: 0;
}

.chat-partner-avatar-message {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  width: 35px;
  height: 35px;
  margin-right: 10px;
  background-color: #ccc;
  color: white;
  font-size: 18px;
  border-radius: 50%;
  text-align: center;
}
</style>
