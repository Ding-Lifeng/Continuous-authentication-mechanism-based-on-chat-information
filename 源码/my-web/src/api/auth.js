import request from '@/utils/request';

export function checkUserStatus() {
    return request({
        url: '/Authentication/checkStatus',
        method: 'get',
    });
}