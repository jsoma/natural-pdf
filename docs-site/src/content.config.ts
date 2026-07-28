import { defineCollection } from "astro:content";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";

// Standard Starlight setup with the default content location
// (src/content/docs/). That directory is gitignored — the staging script
// (scripts/docs_stage.py) populates it from docs/ + docs-executed/.
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
