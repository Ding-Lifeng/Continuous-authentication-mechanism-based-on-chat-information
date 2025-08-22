import request from '@/utils/request'

export function getUserInfo() {
    return request({
        url: '/user/getInfo',
        method: 'get',
    })
}

export function getChatPartners() {
    return request({
        url: '/chatRelation/getChatPartners',
        method: 'get',
    });
}