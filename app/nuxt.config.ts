import vuetify, { transformAssetUrls } from 'vite-plugin-vuetify'

export default defineNuxtConfig({
  devtools: { enabled: true },
  typescript: { strict: true },
  runtimeConfig: {
    public: {
      stacRoot: process.env.NUXT_STAC_ROOT,
      mapboxToken: process.env.NUXT_PUBLIC_MAPBOX_TOKEN,
      pdfEndpoint: process.env.NUXT_PUBLIC_PDF_ENDPOINT,
    },
  },
  css: ['vuetify/styles'],
  modules: [
    (_options, nuxt) => {
      nuxt.hooks.hook('vite:extendConfig', (config) => {
        config.plugins?.push(vuetify({ autoImport: true }))
      })
    },
    '@nuxtjs/tailwindcss',
  ],
  vite: {
    server: {
      fs: {
        allow: ['C:/SnapVolumesTemp/MountPoints/', '..', '.'],
      },
    },
    vue: {
      template: {
        transformAssetUrls,
      },
    },
    ssr: {
      noExternal: ['vuetify'],
    },
  },
  build: {
    transpile: [/echarts/, /zrender/, /tslib/, /resize-detector/],
  },
  serverHandlers:
    process.env.NODE_ENV === 'development'
      ? [
          {
            route: '/stac/**',
            handler: '~/dev/stac/[...].ts',
          },
        ]
      : [],
})
