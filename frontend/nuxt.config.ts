// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  runtimeConfig: {
    public: {
      backendBaseUrl: process.env.NUXT_PUBLIC_BACKEND_BASE_URL || 'http://127.0.0.1:8000',
    },
  },
  modules: ['vuetify-nuxt-module'],
  vuetify: {
    moduleOptions: {},
    vuetifyOptions: {},
  },
})
