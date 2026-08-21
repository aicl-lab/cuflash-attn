import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// Dynamic base path for GitHub Pages deployment
const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/cuflash-attn/'  // fallback for local dev

const head = [
  ['meta', { name: 'theme-color', content: '#76B900' }],
  ['meta', { property: 'og:type', content: 'website' }],
  ['meta', { property: 'og:site_name', content: 'CuFlash-Attn' }],
  ['link', { rel: 'icon', href: '/favicon.svg', type: 'image/svg+xml' }],
  ['link', { rel: 'alternate icon', href: '/favicon.ico', type: 'image/png', sizes: '16x16' }],
  ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
  ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
  ['link', {
    href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;600;700&display=swap',
    rel: 'stylesheet'
  }]
]

const nav = [
  { text: '指南', link: '/guide/quick-start', activeMatch: '/guide/' },
  { text: '深入', link: '/design/kernel-deep-dive', activeMatch: '/design/' },
  { text: '性能', link: '/performance/benchmarks', activeMatch: '/performance/' },
  { text: 'API', link: '/api-reference', activeMatch: '/api-reference' },
  { text: '研究', link: '/research/related-work', activeMatch: '/research/' },
  {
    text: '项目',
    items: [
      { text: '项目状态', link: '/project-status' },
      { text: '发布版本', link: 'https://github.com/open-infra-ai/cuflash-attn/releases' },
      { text: '仓库源码', link: 'https://github.com/open-infra-ai/cuflash-attn' }
    ]
  }
]

const sidebar = [
  {
    text: '开始',
    collapsed: false,
    items: [
      { text: '概览', link: '/' },
      { text: '快速开始', link: '/guide/quick-start' },
      { text: '从源码构建', link: '/building' }
    ]
  },
  {
    text: '架构',
    collapsed: false,
    items: [
      { text: '系统架构', link: '/architecture' },
      { text: '算法详解', link: '/algorithm' },
      { text: 'Kernel 逐行解读', link: '/design/kernel-deep-dive' },
      { text: '设计决策', link: '/design/design-decisions' },
      { text: 'Tensor Core 迁移计划', link: '/design/tensor-core-migration' }
    ]
  },
  {
    text: '性能',
    collapsed: false,
    items: [
      { text: '基准测试', link: '/performance/benchmarks' },
      { text: 'Roofline 分析', link: '/performance/roofline-analysis' }
    ]
  },
  {
    text: '参考',
    collapsed: false,
    items: [
      { text: 'API 参考', link: '/api-reference' },
      { text: '故障排除', link: '/troubleshooting' }
    ]
  },
  {
    text: '研究',
    collapsed: false,
    items: [
      { text: '相关工作', link: '/research/related-work' },
      { text: '参考文献', link: '/research/references' }
    ]
  },
  {
    text: '项目',
    collapsed: false,
    items: [
      { text: '项目状态', link: '/project-status' }
    ]
  }
]

export default withMermaid(defineConfig({
  base,
  title: 'CuFlash-Attn',
  titleTemplate: ':title | CuFlash-Attn',
  description: '从零实现的 CUDA FlashAttention 参考实现',
  lang: 'zh-CN',
  head,

  themeConfig: {
    logo: {
      light: '/logo-light.svg',
      dark: '/logo-dark.svg',
      alt: 'CuFlash-Attn'
    },
    siteTitle: 'CuFlash-Attn',
    nav,
    sidebar,
    outline: { label: '本页目录', level: 'deep' },
    docFooter: { prev: '上一页', next: '下一页' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/open-infra-ai/cuflash-attn' }
    ],
    footer: {
      message: '稳定 v0.5.0 基线 · 精简 CUDA FlashAttention 参考实现',
      copyright: 'Copyright 2026 open-infra-ai.'
    },
    editLink: {
      pattern: 'https://github.com/open-infra-ai/cuflash-attn/edit/master/docs/:path',
      text: '在 GitHub 上编辑此页面'
    },
    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    },
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '外观',
    search: {
      provider: 'local'
    }
  },

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true,
    math: true
  },

  vite: {
    resolve: {
      alias: {
        '@': '/.vitepress'
      }
    }
  },

  srcDir: '.',
  srcExclude: ['**/(README|CHANGELOG|LICENSE|package)*'],
  lastUpdated: true,
  cleanUrls: true
}))
