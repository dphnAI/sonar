# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import html
from pathlib import Path
from urllib.parse import quote


def canonicalize_url(url: str) -> str:
    """Encode path characters that object-storage rewrites may reinterpret."""
    return quote(url, safe=":/%?&=#")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry-file",
        type=Path,
        required=True,
        help="Tab-separated wheel name, URL, and optional SHA-256 digest",
    )
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Sonar nightly wheels")
    parser.add_argument(
        "--description",
        default="Precompiled CUDA wheels from every commit on main. "
        "These are development snapshots and may change without notice.",
    )
    parser.add_argument(
        "--install-command",
        default="uv pip install sonar "
        "--extra-index-url https://sonar.dphn.ai/whl/nightly/cuda/x86_64/simple "
        "--index-strategy first-index --torch-backend cu132",
    )
    args = parser.parse_args()

    cards = []
    for line_number, line in enumerate(args.entry_file.read_text().splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) not in (2, 3):
            raise ValueError(f"{args.entry_file}:{line_number}: expected 2 or 3 tab-separated fields")
        name, url = fields[:2]
        digest = fields[2] if len(fields) == 3 else ""
        if not name.endswith(".whl") or Path(name).name != name:
            raise ValueError(f"Invalid wheel name: {name!r}")
        # Shared storage may still contain wheels from before the package rename.
        # Never advertise another distribution in Sonar's simple package index.
        if not name.startswith("sonar-"):
            continue
        url = canonicalize_url(url)
        if digest:
            url = f"{url}#sha256={digest}"
        try:
            version, python, abi, platform = name.removeprefix("sonar-").removesuffix(".whl").rsplit("-", 3)
            platform = platform.replace("_", " ", 1)
        except ValueError:
            version, python, abi, platform = "Nightly build", "Python", "ABI", "Linux"
        cards.append(
            f"""      <article class="wheel" data-search="{html.escape(version.lower(), quote=True)}">
        <div class="wheel-main">
          <strong>{html.escape(version)}</strong>
          <div class="tags">
            <span>{html.escape(python)}</span><span>{html.escape(abi)}</span><span>{html.escape(platform)}</span>
          </div>
        </div>
        <a class="download" href="{html.escape(url, quote=True)}"
           data-requires-python="&gt;=3.11,&lt;3.15"
           aria-label="Download Sonar {html.escape(version, quote=True)}">
          <span class="filename">{html.escape(name)}</span><span class="icon" aria-hidden="true">↓</span>
        </a>
      </article>"""
        )

    if not cards:
        raise ValueError(f"No wheel entries found in {args.entry_file}")
    escaped_commit = html.escape(args.commit)
    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(args.title)}</title>
    <meta name="description" content="{html.escape(args.description, quote=True)}">
    <style>
      :root {{
        color-scheme: light dark;
        --bg: #fff; --text: #1f2328; --muted: #656d76;
        --line: #d8dee4; --row: #f6f8fa; --accent: #0969da;
      }}
      * {{ box-sizing: border-box }}
      body {{
        margin: 0; color: var(--text); background: var(--bg);
        font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
      }}
      main {{ width: min(960px, calc(100% - 32px)); margin: auto; padding: 48px 0 72px }}
      header {{ padding-bottom: 24px; border-bottom: 1px solid var(--line) }}
      .eyebrow {{ color: var(--muted); font-size: 13px }}
      h1 {{ margin: 4px 0 0; font-size: 26px; font-weight: 600; letter-spacing: -.02em }}
      .lede {{ margin: 8px 0 0; color: var(--muted) }}
      .install {{
        display: flex; align-items: center; gap: 8px; max-width: 680px;
        margin-top: 20px; padding: 7px 7px 7px 12px;
        border: 1px solid var(--line); border-radius: 6px; background: var(--row);
      }}
      code {{
        flex: 1; overflow-x: auto; color: inherit;
        font: 12px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: nowrap;
      }}
      button, input {{
        color: var(--text); border: 1px solid var(--line); border-radius: 6px;
        outline: none; background: var(--bg);
      }}
      button {{ padding: 5px 10px; cursor: pointer }}
      button:hover {{ background: var(--row) }}
      input {{ width: 250px; margin-left: auto; padding: 7px 10px }}
      input:focus {{
        border-color: var(--accent);
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 20%, transparent);
      }}
      .toolbar {{ display: flex; align-items: center; gap: 12px; margin: 28px 0 10px }}
      h2 {{ margin: 0; font-size: 14px; font-weight: 600 }}
      .count {{ color: var(--muted); font-size: 12px }}
      .wheels {{
        overflow: hidden; border: 1px solid var(--line); border-radius: 7px;
      }}
      .wheel {{
        display: grid; grid-template-columns: minmax(220px, .9fr) minmax(0, 1.5fr);
        align-items: center; border-bottom: 1px solid var(--line);
      }}
      .wheel:last-child {{ border-bottom: 0 }}
      .wheel:hover {{ background: var(--row) }}
      .wheel-main {{ min-width: 0; padding: 11px 14px }}
      .wheel-main strong {{ display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap }}
      .tags {{ display: flex; gap: 10px; margin-top: 2px }}
      .tags span {{
        color: var(--muted); font: 11px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
      }}
      .download {{
        display: flex; min-width: 0; align-self: stretch; align-items: center; gap: 10px; padding: 11px 14px;
        color: var(--muted); border-left: 1px solid var(--line); text-decoration: none;
      }}
      .download:hover {{ color: var(--accent) }}
      .filename {{
        overflow: hidden; font: 11px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
        text-overflow: ellipsis; white-space: nowrap;
      }}
      .icon {{
        margin-left: auto; color: var(--muted); font-size: 16px;
      }}
      footer {{ margin-top: 18px; color: var(--muted); font-size: 12px }}
      footer a {{ color: var(--accent) }}
      .empty {{ display: none; padding: 28px; color: var(--muted); text-align: center }}
      .pagination {{
        display: flex; align-items: center; justify-content: flex-end; gap: 8px; margin-top: 12px;
      }}
      .pagination span {{ min-width: 120px; color: var(--muted); font-size: 12px; text-align: center }}
      button:disabled {{ color: var(--muted); cursor: default; opacity: .55 }}
      button:disabled:hover {{ background: var(--bg) }}
      [hidden] {{ display: none }}
      @media (max-width: 700px) {{
        main {{ padding-top: 28px }} .toolbar {{ align-items: stretch; flex-direction: column }}
        input {{ width: 100%; margin-left: 0 }} .wheel {{ grid-template-columns: 1fr }}
        .download {{ border-top: 1px solid var(--line); border-left: 0 }}
      }}
      @media (prefers-color-scheme: dark) {{
        :root {{
          --bg: #111315; --text: #e6e8eb; --muted: #9299a1;
          --line: #30363d; --row: #191c20; --accent: #58a6ff;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <div class="eyebrow">sonar.dphn.ai / whl</div>
        <h1>{html.escape(args.title)}</h1>
        <p class="lede">{html.escape(args.description)}</p>
        <div class="install">
          <code id="command">{html.escape(args.install_command)}</code>
          <button id="copy" type="button">Copy</button>
        </div>
        <p class="lede">Use the Sonar index in the command above, not a plain PyPI install of <code>sonar</code>.</p>
      </header>
      <div class="toolbar">
        <div><h2>Available builds</h2><span class="count" id="count">{len(cards)} wheels</span></div>
        <input id="filter" type="search" placeholder="Filter by version or commit…" aria-label="Filter wheels">
      </div>
      <section class="wheels" aria-label="Available wheels">
{chr(10).join(cards)}
      </section>
      <p class="empty" id="empty">No wheels match that filter.</p>
      <nav class="pagination" id="pagination" aria-label="Wheel pages">
        <button id="previous" type="button">Previous</button>
        <span id="range"></span>
        <button id="next" type="button">Next</button>
      </nav>
      <footer>Latest index build:
        <a href="https://github.com/dphnAI/sonar/commit/{escaped_commit}"><code>{escaped_commit[:12]}</code></a>
        · Requires Python 3.11–3.14
      </footer>
    </main>
    <!-- Latest build from {escaped_commit} -->
    <script>
      const wheels = [...document.querySelectorAll(".wheel")];
      const filter = document.querySelector("#filter");
      const count = document.querySelector("#count");
      const empty = document.querySelector("#empty");
      const pagination = document.querySelector("#pagination");
      const previous = document.querySelector("#previous");
      const next = document.querySelector("#next");
      const range = document.querySelector("#range");
      const pageSize = 50;
      let page = 0;

      function render() {{
        const query = filter.value.trim().toLowerCase();
        const matches = wheels.filter(wheel => wheel.dataset.search.includes(query));
        const pageCount = Math.max(1, Math.ceil(matches.length / pageSize));
        page = Math.min(page, pageCount - 1);
        const start = page * pageSize;
        const visible = new Set(matches.slice(start, start + pageSize));
        for (const wheel of wheels) {{
          wheel.hidden = !visible.has(wheel);
        }}
        count.textContent = matches.length === wheels.length
          ? wheels.length + " wheels"
          : matches.length + " of " + wheels.length + " wheels";
        empty.style.display = matches.length ? "none" : "block";
        pagination.hidden = matches.length <= pageSize;
        previous.disabled = page === 0;
        next.disabled = page >= pageCount - 1;
        range.textContent = matches.length
          ? (start + 1) + "–" + Math.min(start + pageSize, matches.length) + " of " + matches.length
          : "0 wheels";
      }}

      filter.addEventListener("input", () => {{
        page = 0;
        render();
      }});
      previous.addEventListener("click", () => {{
        page -= 1;
        render();
        document.querySelector(".toolbar").scrollIntoView();
      }});
      next.addEventListener("click", () => {{
        page += 1;
        render();
        document.querySelector(".toolbar").scrollIntoView();
      }});
      document.querySelector("#copy").addEventListener("click", async event => {{
        await navigator.clipboard.writeText(document.querySelector("#command").textContent);
        event.currentTarget.textContent = "Copied";
        setTimeout(() => event.currentTarget.textContent = "Copy", 1500);
      }});
      render();
    </script>
  </body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)


if __name__ == "__main__":
    main()
