/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Server Connection
  readonly VITE_SERVER_HOST?: string;
  readonly VITE_HTTPS?: string;

  // Language & Locale
  readonly VITE_DEFAULT_LANGUAGE?: 'en' | 'ko' | 'auto';

  // Font & Typography
  // Korean: default, noto-sans, casual, elegant, nanum, handwriting, playful
  // English: courier, consolas, mono
  readonly VITE_DEFAULT_FONT?: 'default' | 'noto-sans' | 'casual' | 'elegant' | 'nanum' | 'handwriting' | 'playful' | 'courier' | 'consolas' | 'mono';
  readonly VITE_DEFAULT_FONT_SIZE?: string;

  // Realtime STT Display
  readonly VITE_MAX_WORDS_PER_LINE?: string;

  // Theme
  readonly VITE_DEFAULT_THEME?: 'dark' | 'light';
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
