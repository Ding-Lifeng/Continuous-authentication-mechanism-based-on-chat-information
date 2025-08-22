import request from '@/utils/request';

// 获取当前时间，精确到分钟，并转换为中国时间
function getChinaTime() {
    const now = new Date();
    now.setHours(now.getHours());
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');

    return `${year}-${month}-${day}T${hours}:${minutes}`;
}

// 发送聊天内容
export function sendChatContent(chatPartnerId, messageContent) {
    const data = {
        time: getChinaTime(), // 获取当前时间，精确到分钟
        chatPartner_id: chatPartnerId,       // 聊天对象的id
        content: messageContent // 用户输入的聊天消息
    };
    return request({
        url: '/chatContent/add',
        method: 'post',
        data: data
    });
}

// 获取与指定聊天对象的聊天记录
export function getChatContent(chatPartnerId) {
    return request({
        url: '/chatContent/get',  // 后端接口地址
        method: 'get',
        params: { chatPartnerId }, // 使用查询参数传递 chatPartnerId
    });
}


// 删除聊天记录
export function deleteChatContent(messageId) {
    const data = {
        message_id: messageId
    };
    return request({
        url: '/chatContent/delete',  // 后端接口地址，删除聊天消息
        method: 'post',
        data: data
    });
}
