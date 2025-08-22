import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import './permission'
import VEmojiPicker from 'v-emoji-picker';

Vue.use(ElementUI);

Vue.config.productionTip = false

Vue.use(VEmojiPicker);

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
