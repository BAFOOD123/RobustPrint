// Turns dist-artifact/index.html into the claude.ai Artifact page: no document wrapper,
// React 18.3.1 from cdnjs, fonts from Google Fonts, everything else inline.
import { readFileSync, writeFileSync } from "node:fs";

const html = readFileSync("dist-artifact/index.html", "utf8");
const grab = (re, what) => {
  const m = html.match(re);
  if (!m) throw new Error(`artifact: ${what} not found in the build`);
  return m[1];
};
const style = grab(/<style[^>]*>([\s\S]*?)<\/style>/, "inline style");
const script = grab(/<script type="module"[^>]*>([\s\S]*?)<\/script>/, "inline module script");
const fonts = grab(/(<link href="https:\/\/fonts\.googleapis\.com[^>]*>)/, "font link");
const CDN = "https://cdnjs.cloudflare.com/ajax/libs";
const page = [
  "<title>RobustPrint Virtual Lab</title>",
  '<meta name="claude:viewport-fit-sides" content="cover">',
  fonts,
  `<script src="${CDN}/react/18.3.1/umd/react.production.min.js" crossorigin="anonymous"></script>`,
  `<script src="${CDN}/react-dom/18.3.1/umd/react-dom.production.min.js" crossorigin="anonymous"></script>`,
  `<style>${style}</style>`,
  '<div id="root"></div>',
  `<script type="module">${script.replace(/<\/script/gi, "<\\/script")}</script>`,
].join("\n");
writeFileSync("dist-artifact/artifact.html", page);
console.log(`artifact.html: ${(page.length / 1024).toFixed(0)} KB`);
