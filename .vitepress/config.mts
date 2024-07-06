import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Prompt Engineering Chinese Translation",
  description:
    "Tips and tricks for working with Large Language Models like OpenAI's GPT-4.",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      {
        text: "Guide",
        link: "/guide/what-is-a-large-language-model-llm",
        activeMatch: "/guide/",
      },
      { text: "Reference", link: "/README" },
    ],
    sidebar: {
      "/guide/": {
        base: "/guide/",
        items: [
          {
            text: "Prompt Engineering",
            items: [
              {
                text: "What is a Large Language Model (LLM)?",
                link: "what-is-a-large-language-model-llm",
              },
              {
                text: "What is a prompt?",
                link: "what-is-a-prompt",
              },
              {
                text: "Why do we need prompt engineering?",
                link: "why-do-we-need-prompt-engineering",
              },
              {
                text: "Strategies",
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
  },
});
