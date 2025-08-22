import request from '@/utils/request';

// 获取聊天对象信息
export function getChatPartnerInfo(chatPartnerId) {
    return request({
        url: `/user/get/${chatPartnerId}`,
        method: 'get',
    });
}