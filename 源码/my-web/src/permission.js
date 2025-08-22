import router from './router'
import { getAccessToken, removeToken } from '@/utils/auth'
import { checkUserStatus } from '@/api/auth'

router.beforeEach(async (to, from, next) => {
  const token = getAccessToken();

  if (token) { // 如果有token
    try {
      // 每次路由跳转时检查用户状态
      const response = await checkUserStatus();
      if (response.data === "banned") {
        removeToken();
        next({ path: '/login' });
        return;
      }
    } catch (error) {
      console.error('状态检查失败:', error);
    }

    if (to.path === '/login') {
      next({ path: '/' }) // 如果是去登录页，重定向到首页
    } else {
      next()  // 正常跳转
    }
  } else { // 没有token
    if (to.path === '/login' || to.path === '/submit') {
      next();
    } else {
      next({path: '/login'});
    }
  }
})

router.afterEach(() => {
  // 这里可以添加一些在路由跳转后需要执行的代码
})
