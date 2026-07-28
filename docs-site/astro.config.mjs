// Astro + Starlight configuration for the Natural PDF docs site.
//
// Content in src/content/docs/ and static files in public/ are populated by
// scripts/docs_stage.py (both directories are gitignored). This file is the
// single source of truth for site URL, sidebar order, and legacy redirects
// (ported verbatim from mkdocs.yml).
import { defineConfig, passthroughImageService } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkNpdfTabs from "./src/plugins/remark-npdf-tabs.mjs";

export default defineConfig({
  site: "https://jsoma.github.io",
  base: "/natural-pdf",
  trailingSlash: "always",
  build: { format: "directory" },
  image: {
    // Generated screenshots must pass through untouched — never optimized,
    // never re-encoded (WebP "optimization" can enlarge them). This also
    // means no sharp dependency.
    service: passthroughImageService(),
  },
  vite: {
    // Never inline assets/scripts as data URIs or inline <script> blocks:
    // the tabs script and favicon should ship as hashed files in _astro/.
    build: { assetsInlineLimit: 0 },
  },
  markdown: {
    remarkPlugins: [remarkNpdfTabs],
  },
  integrations: [
    starlight({
      title: "Natural PDF",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/jsoma/natural-pdf",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      // No right-hand "On this page" panel.
      tableOfContents: false,
      // Wrap long code lines instead of horizontal scrolling.
      expressiveCode: {
        defaultProps: { wrap: true },
      },
      // The default favicon link resolves to public/favicon.svg, which the
      // staging script provides. src/components/Head.astro additionally emits
      // a Vite-bundled copy of the same icon so the site never lacks one.
      favicon: "/favicon.svg",
      components: {
        Head: "./src/components/Head.astro",
      },
      // Sidebar ported 1:1 from the `nav:` section of mkdocs.yml
      // (35 pages; group labels, page labels, and order verbatim).
      sidebar: [
        { label: "Home", link: "/" },
        {
          label: "Get Started",
          items: [
            { slug: "get-started" },
            { label: "Quickstart", slug: "get-started/quickstart" },
            { label: "Coming from pdfplumber", slug: "get-started/from-pdfplumber" },
          ],
        },
        {
          label: "Learn",
          items: [
            { slug: "learn" },
            { label: "1. Text and tables", slug: "learn/01-text-and-tables" },
            { label: "2. OCR", slug: "learn/02-ocr" },
            { label: "3. AI extraction", slug: "learn/03-ai-extraction" },
            { label: "4. Page structure", slug: "learn/04-page-structure" },
            { label: "5. Putting it together", slug: "learn/05-putting-it-together" },
          ],
        },
        {
          label: "Concepts",
          items: [
            { slug: "concepts" },
            { label: "The spatial model", slug: "concepts/spatial-model" },
            { label: "How text becomes elements", slug: "concepts/text-and-elements" },
            { label: "Exclusions", slug: "concepts/exclusions" },
            { label: "Selectors", slug: "concepts/selectors" },
            { label: "Tables, a decision guide", slug: "concepts/tables" },
            { label: "Engines and models", slug: "concepts/engines-and-models" },
          ],
        },
        {
          label: "Solve",
          items: [
            { slug: "solve" },
            { label: "Multi-column reflow", slug: "solve/multi-column-reflow" },
            { label: "Zebra-stripe table", slug: "solve/zebra-stripe-table" },
            { label: "Arabic election table", slug: "solve/arabic-election-table" },
            { label: "Use-of-force logs", slug: "solve/use-of-force-logs" },
            { label: "Pixelated scan table", slug: "solve/pixelated-scan-table" },
            { label: "Serbian multi-page table", slug: "solve/serbian-multipage-table" },
            { label: "Borderless call log", slug: "solve/borderless-call-log" },
            { label: "Complaint database printout", slug: "solve/complaint-database-printout" },
          ],
        },
        { label: "Troubleshooting", slug: "troubleshooting" },
        {
          label: "Reference",
          items: [
            { label: "Selectors", slug: "reference/selectors" },
            { label: "Engines", slug: "reference/engines" },
            { label: "Installation extras", slug: "reference/installation-extras" },
            { label: "Exceptions", slug: "reference/exceptions" },
            { label: "OCR options", slug: "reference/ocr-options" },
            { label: "API", slug: "api" },
            { label: "Text Extraction", slug: "api/text-extraction" },
          ],
        },
        { label: "For Agents", slug: "for-agents" },
      ],
    }),
  ],
  // Legacy URL redirects, ported 1:1 from the redirect_maps in mkdocs.yml
  // (59 entries). Astro requires sources WITHOUT the base prefix but
  // destinations WITH it — a destination without /natural-pdf 404s.
  redirects: {
    // Retired getting-started section -> new six-section site
    "/installation/": "/natural-pdf/get-started/",
    "/getting-started/": "/natural-pdf/get-started/",
    "/getting-started/quickstart/": "/natural-pdf/get-started/quickstart/",
    "/getting-started/selectors/": "/natural-pdf/concepts/selectors/",
    "/getting-started/concepts/": "/natural-pdf/concepts/",
    "/getting-started/choose-your-path/": "/natural-pdf/learn/",
    // Retired tutorials -> Learn / Concepts
    "/tutorials/01-loading-and-extraction/": "/natural-pdf/learn/01-text-and-tables/",
    "/tutorials/02-finding-elements/": "/natural-pdf/learn/01-text-and-tables/",
    "/tutorials/03-extracting-blocks/": "/natural-pdf/concepts/spatial-model/",
    "/tutorials/04-table-extraction/": "/natural-pdf/concepts/tables/",
    "/tutorials/05-excluding-content/": "/natural-pdf/concepts/exclusions/",
    "/tutorials/06-document-qa/": "/natural-pdf/learn/03-ai-extraction/",
    "/tutorials/07-layout-analysis/": "/natural-pdf/learn/04-page-structure/",
    "/tutorials/07-working-with-regions/": "/natural-pdf/learn/04-page-structure/",
    "/tutorials/08-spatial-navigation/": "/natural-pdf/concepts/spatial-model/",
    "/tutorials/09-section-extraction/": "/natural-pdf/learn/04-page-structure/",
    "/tutorials/10-form-field-extraction/": "/natural-pdf/learn/03-ai-extraction/",
    "/tutorials/11-enhanced-table-processing/": "/natural-pdf/concepts/tables/",
    "/tutorials/12-ocr-integration/": "/natural-pdf/learn/02-ocr/",
    "/tutorials/13-semantic-search/": "/natural-pdf/learn/03-ai-extraction/",
    "/tutorials/14-categorizing-documents/": "/natural-pdf/learn/03-ai-extraction/",
    "/tutorials/15-working-with-regions/": "/natural-pdf/learn/04-page-structure/",
    // Retired cookbook -> Learn / Concepts / Solve / Troubleshooting
    "/cookbook/": "/natural-pdf/learn/05-putting-it-together/",
    "/cookbook/batch-processing/": "/natural-pdf/learn/05-putting-it-together/",
    "/cookbook/one-page-one-row/": "/natural-pdf/learn/05-putting-it-together/",
    "/cookbook/finding-sections/": "/natural-pdf/learn/05-putting-it-together/",
    "/cookbook/structured-extraction/": "/natural-pdf/learn/03-ai-extraction/",
    "/cookbook/label-value-extraction/": "/natural-pdf/learn/01-text-and-tables/",
    "/cookbook/guides/": "/natural-pdf/concepts/tables/",
    "/cookbook/messy-tables/": "/natural-pdf/concepts/tables/",
    "/cookbook/form-cells/": "/natural-pdf/concepts/tables/",
    "/cookbook/multi-column-layouts/": "/natural-pdf/solve/multi-column-reflow/",
    "/cookbook/ocr-then-navigate/": "/natural-pdf/learn/02-ocr/",
    "/cookbook/multipage-content/": "/natural-pdf/learn/04-page-structure/",
    "/cookbook/troubleshooting/": "/natural-pdf/troubleshooting/",
    // Retired standalone sections
    "/use-cases/idea-gallery/": "/natural-pdf/solve/",
    "/quick-reference/": "/natural-pdf/reference/selectors/",
    "/for-llms/common-patterns/": "/natural-pdf/for-agents/",
    "/for-llms/anti-patterns/": "/natural-pdf/for-agents/",
    // Pre-rewrite how-to guide URLs -> new equivalents
    "/text-extraction/": "/natural-pdf/learn/01-text-and-tables/",
    "/ocr/": "/natural-pdf/learn/02-ocr/",
    "/extracting-clean-text/": "/natural-pdf/concepts/exclusions/",
    "/fix-messy-tables/": "/natural-pdf/concepts/tables/",
    "/tables/": "/natural-pdf/concepts/tables/",
    "/process-forms-and-invoices/": "/natural-pdf/concepts/spatial-model/",
    "/data-extraction/": "/natural-pdf/learn/03-ai-extraction/",
    "/document-qa/": "/natural-pdf/learn/03-ai-extraction/",
    "/categorizing-documents/": "/natural-pdf/learn/03-ai-extraction/",
    "/element-selection/": "/natural-pdf/concepts/selectors/",
    "/pdf-navigation/": "/natural-pdf/learn/01-text-and-tables/",
    "/layout-analysis/": "/natural-pdf/learn/04-page-structure/",
    "/regions/": "/natural-pdf/learn/04-page-structure/",
    "/visual-debugging/": "/natural-pdf/learn/01-text-and-tables/",
    "/interactive-widget/": "/natural-pdf/learn/01-text-and-tables/",
    "/describe/": "/natural-pdf/learn/01-text-and-tables/",
    "/loops-and-groups/": "/natural-pdf/learn/05-putting-it-together/",
    "/text-analysis/": "/natural-pdf/concepts/selectors/",
    "/reflowing-pages/": "/natural-pdf/solve/multi-column-reflow/",
    "/finetuning/": "/natural-pdf/learn/02-ocr/",
  },
});
