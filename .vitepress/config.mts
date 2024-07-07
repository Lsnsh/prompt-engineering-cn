import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: "/prompt-engineering-cn/",
  title: "提示词工程指南",
  description: "使用OpenAI的GPT-4等大型语言模型的提示和技巧。",
  cleanUrls: true,
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      {
        text: "指南",
        link: "/guide/what-is-a-large-language-model-llm",
        activeMatch: "/guide/",
      },
      { text: "参考", link: "/README" },
    ],
    sidebar: {
      "/guide/": {
        base: "/guide/",
        items: [
          {
            text: "提示词工程",
            items: [
              {
                text: "什么是大语言模型 (LLM)？",
                link: "what-is-a-large-language-model-llm",
              },
              {
                text: "什么是提示词？",
                link: "what-is-a-prompt",
              },
              {
                text: "为什么我们需要提示词工程？",
                link: "why-do-we-need-prompt-engineering",
              },
              {
                text: "策略",
                link: "strategies",
              },
            ],
          },
        ],
      },
    },
    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/Lsnsh/prompt-engineering-cn",
      },
    ],
    docFooter: {
      prev: "上一页",
      next: "下一页",
    },
    outline: {
      label: "页面导航",
    },
    lastUpdated: {
      text: "最后更新于",
      formatOptions: {
        dateStyle: "short",
        timeStyle: "medium",
      },
    },

    langMenuLabel: "多语言",
    returnToTopLabel: "回到顶部",
    sidebarMenuLabel: "菜单",
    darkModeSwitchLabel: "主题",
    lightModeSwitchTitle: "切换到浅色模式",
    darkModeSwitchTitle: "切换到深色模式",
  },
});
