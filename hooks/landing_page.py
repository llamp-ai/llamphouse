"""
MkDocs hook — generate the product landing page.

After each ``mkdocs build`` this hook writes ``site/index.html`` (one level
above ``site_dir: site/docs``) so that the root of the website shows a custom
landing page while the full documentation lives at ``/docs/``.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Landing page HTML
# ---------------------------------------------------------------------------
LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LLAMPHouse — Serving Your LLM Apps, Scalable and Reliable</title>
  <meta name="description" content="Self-hosted, production-ready server for LLM-powered applications. OpenAI-compatible API, A2A protocol, streaming, multi-agent orchestration, and a built-in observability dashboard." />
  <meta property="og:title" content="LLAMPHouse" />
  <meta property="og:description" content="Self-hosted server for LLM-powered applications. OpenAI Assistants API + A2A protocol out of the box." />
  <meta property="og:image" content="./docs/img/llamphouse.png" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:         #09090b;
      --bg-card:    #18181b;
      --bg-code:    #0f0f11;
      --border:     #27272a;
      --text:       #fafafa;
      --text-muted: #71717a;
      --accent:     #a855f7;
      --accent-dim: rgba(168,85,247,.12);
      --mono: 'Menlo','Monaco','Consolas',ui-monospace,monospace;
      --sans: -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
    }

    @media (prefers-color-scheme: light) {
      :root {
        --bg:         #ffffff;
        --bg-card:    #f4f4f5;
        --bg-code:    #f1f1f3;
        --border:     #e4e4e7;
        --text:       #09090b;
        --text-muted: #71717a;
        --accent:     #7c3aed;
        --accent-dim: rgba(124,58,237,.08);
      }
    }

    html { font-family: var(--sans); scroll-behavior: smooth; }
    body { background: var(--bg); color: var(--text); line-height: 1.65; min-height: 100vh; }

    a { color: inherit; text-decoration: none; }
    code { font-family: var(--mono); font-size: .88em;
           background: var(--bg-card); border: 1px solid var(--border);
           border-radius: 4px; padding: .1em .35em; }

    /* ── Nav ──────────────────────────────────────── */
    nav {
      position: sticky; top: 0; z-index: 50;
      display: flex; align-items: center; justify-content: space-between;
      padding: .9rem 2rem;
      background: color-mix(in srgb, var(--bg) 80%, transparent);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
    }
    .nav-logo {
      display: flex; align-items: center; gap: .5rem;
      font-size: 1rem; font-weight: 700; letter-spacing: -.01em;
    }
    .nav-logo img { width: 28px; height: 28px; object-fit: contain; }
    .nav-links { display: flex; align-items: center; gap: 1.25rem; }
    .nav-links a { font-size: .875rem; color: var(--text-muted); transition: color .15s; }
    .nav-links a:hover { color: var(--text); }
    .nav-gh {
      display: inline-flex; align-items: center; gap: .4rem;
      padding: .4rem .9rem; border-radius: 7px;
      border: 1px solid var(--border); background: var(--bg-card);
      font-size: .825rem; font-weight: 500;
      transition: border-color .15s;
    }
    .nav-gh:hover { border-color: var(--text-muted); }

    /* ── Hero ─────────────────────────────────────── */
    .hero {
      max-width: 760px; margin: 0 auto;
      padding: 7rem 2rem 5rem;
      text-align: center;
    }
    .badge {
      display: inline-flex; align-items: center; gap: .45rem;
      padding: .3rem .85rem; border-radius: 99px;
      border: 1px solid var(--border); background: var(--bg-card);
      font-size: .78rem; color: var(--text-muted); margin-bottom: 2rem;
      letter-spacing: .01em;
    }
    .badge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
    .hero h1 {
      font-size: clamp(2.6rem, 7vw, 4.2rem);
      font-weight: 800; line-height: 1.1;
      letter-spacing: -.035em; margin-bottom: 1.4rem;
    }
    .hero h1 mark {
      background: none; color: var(--accent);
      -webkit-background-clip: text;
    }
    .hero p {
      font-size: 1.125rem; color: var(--text-muted);
      max-width: 520px; margin: 0 auto 2.5rem;
    }
    .hero-btns {
      display: flex; gap: .75rem; justify-content: center;
      flex-wrap: wrap; margin-bottom: 2.25rem;
    }
    .btn-primary {
      display: inline-flex; align-items: center; gap: .4rem;
      padding: .65rem 1.4rem; border-radius: 8px;
      background: var(--accent); color: #fff;
      font-size: .9rem; font-weight: 600;
      transition: opacity .15s;
    }
    .btn-primary:hover { opacity: .88; }
    .btn-outline {
      display: inline-flex; align-items: center; gap: .4rem;
      padding: .65rem 1.4rem; border-radius: 8px;
      border: 1px solid var(--border); background: var(--bg-card);
      font-size: .9rem; font-weight: 500;
      transition: border-color .15s;
    }
    .btn-outline:hover { border-color: var(--text-muted); }
    .install {
      display: inline-flex; align-items: center; gap: .75rem;
      background: var(--bg-card); border: 1px solid var(--border);
      border-radius: 9px; padding: .6rem 1.25rem;
      font-family: var(--mono); font-size: .875rem; color: var(--text-muted);
      cursor: default;
    }
    .install-prompt { color: var(--accent); user-select: none; }
    .install-cmd { color: var(--text); }

    /* ── Feature grid ─────────────────────────────── */
    .features-wrap {
      max-width: 1080px; margin: 0 auto; padding: 0 2rem 5rem;
    }
    .section-label {
      text-align: center; margin-bottom: 2.5rem;
    }
    .section-label h2 {
      font-size: 1.75rem; font-weight: 700; letter-spacing: -.025em; margin-bottom: .4rem;
    }
    .section-label p { color: var(--text-muted); font-size: .95rem; }
    .features {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      border: 1px solid var(--border); border-radius: 14px;
      overflow: hidden;
      background: var(--border);  /* gap colour */
      gap: 1px;
    }
    .feature {
      padding: 1.75rem; background: var(--bg);
      transition: background .2s;
    }
    .feature:hover { background: var(--bg-card); }
    .feat-icon {
      width: 38px; height: 38px; border-radius: 9px;
      background: var(--accent-dim); border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
      display: flex; align-items: center; justify-content: center;
      font-size: 1.1rem; margin-bottom: .9rem;
    }
    .feature h3 { font-size: .925rem; font-weight: 600; margin-bottom: .35rem; }
    .feature p { font-size: .83rem; color: var(--text-muted); line-height: 1.55; }

    /* ── Code snippet ─────────────────────────────── */
    .code-wrap {
      max-width: 800px; margin: 0 auto; padding: 0 2rem 5rem;
    }
    .code-wrap .section-label { text-align: left; }
    .file-tab {
      display: inline-block;
      background: var(--bg-card); border: 1px solid var(--border);
      border-bottom: none; border-radius: 6px 6px 0 0;
      padding: .25rem .7rem;
      font-family: var(--mono); font-size: .75rem; color: var(--text-muted);
    }
    .file-tab + pre { border-radius: 0 12px 12px 12px; }
    pre {
      background: var(--bg-card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.5rem 1.75rem;
      overflow-x: auto; line-height: 1.75;
      font-family: var(--mono); font-size: .825rem;
    }
    /* syntax colours */
    .kw  { color: #c084fc; }   /* keyword */
    .cl  { color: #67e8f9; }   /* class  */
    .fn  { color: #86efac; }   /* function */
    .st  { color: #fde68a; }   /* string */
    .cm  { color: #52525b; }   /* comment */
    .im  { color: #a5b4fc; }   /* module */
    .nb  { color: #fb923c; }   /* number */
    @media (prefers-color-scheme: light) {
      .kw  { color: #7c3aed; }
      .cl  { color: #0891b2; }
      .fn  { color: #166534; }
      .st  { color: #b45309; }
      .cm  { color: #a1a1aa; }
      .im  { color: #4338ca; }
      .nb  { color: #c2410c; }
    }

    /* ── CTA banner ───────────────────────────────── */
    .cta-wrap {
      max-width: 1080px; margin: 0 auto; padding: 0 2rem 6rem;
    }
    .cta {
      border: 1px solid var(--border); border-radius: 16px;
      background: var(--bg-card);
      padding: 3.5rem 2rem; text-align: center;
      background-image: radial-gradient(ellipse at 50% 0%, var(--accent-dim), transparent 70%);
    }
    .cta h2 {
      font-size: 1.9rem; font-weight: 700; letter-spacing: -.025em;
      margin-bottom: .6rem;
    }
    .cta p { color: var(--text-muted); margin-bottom: 2rem; font-size: .95rem; }

    /* ── Footer ───────────────────────────────────── */
    footer {
      border-top: 1px solid var(--border);
      padding: 1.75rem 2rem;
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 1rem;
      font-size: .8rem; color: var(--text-muted);
    }
    .footer-links { display: flex; gap: 1.25rem; flex-wrap: wrap; }
    .footer-links a { color: var(--text-muted); transition: color .15s; }
    .footer-links a:hover { color: var(--text); }

    @media (max-width: 640px) {
      nav { padding: .8rem 1.25rem; }
      .nav-links span { display: none; }
      .hero { padding: 4rem 1.5rem 3.5rem; }
      footer { flex-direction: column; text-align: center; }
    }
  </style>
</head>
<body>

  <!-- ── Nav ──────────────────────────────────────── -->
  <nav>
    <a href="." class="nav-logo">
      <img src="./docs/img/llamphouse.png" alt="LLAMPHouse logo" />
      <span>LLAMPHouse</span>
    </a>
    <div class="nav-links">
      <a href="./docs/" class="nav-link">Docs</a>
      <a href="./docs/examples/" class="nav-link">Examples</a>
      <a href="https://github.com/llamp-ai/llamphouse" class="nav-gh" target="_blank" rel="noopener">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 .3C5.37.3 0 5.67 0 12.3c0 5.3 3.44 9.8 8.2 11.4.6.1.83-.26.83-.57v-2c-3.34.72-4.04-1.61-4.04-1.61-.54-1.38-1.33-1.75-1.33-1.75-1.08-.74.08-.72.08-.72 1.2.08 1.83 1.23 1.83 1.23 1.06 1.82 2.8 1.3 3.48.99.1-.77.41-1.3.75-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.3.47-2.38 1.24-3.22-.13-.3-.54-1.52.11-3.17 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.65 1.65.24 2.87.12 3.17.77.84 1.23 1.92 1.23 3.22 0 4.61-2.8 5.63-5.48 5.92.43.37.82 1.1.82 2.22v3.29c0 .32.22.68.83.56C20.56 22.1 24 17.6 24 12.3 24 5.67 18.63.3 12 .3z"/>
        </svg>
        GitHub
      </a>
    </div>
  </nav>

  <main>

    <!-- ── Hero ───────────────────────────────────── -->
    <section class="hero">
      <div class="badge">
        <span class="badge-dot"></span>
        A2A Protocol support — v1.2.0
      </div>
      <h1>Serve your LLM agents,<br><mark>production-ready</mark></h1>
      <p>Self-hosted server for LLM-powered apps. OpenAI-compatible Assistants API, A2A agent-to-agent protocol, and full observability — in a single <code>pip install</code>.</p>
      <div class="hero-btns">
        <a href="./docs/getting-started/quickstart/" class="btn-primary">
          Get Started
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
        </a>
        <a href="./docs/" class="btn-outline">
          Read the Docs
        </a>
      </div>
      <div class="install">
        <span class="install-prompt">$</span>
        <span class="install-cmd">pip install llamphouse</span>
      </div>
    </section>

    <!-- ── Features ───────────────────────────────── -->
    <div class="features-wrap">
      <div class="section-label">
        <h2>Everything you need to ship agents</h2>
        <p>LLAMPHouse focuses on <em>serving</em> agents — not just building them.</p>
      </div>
      <div class="features">
        <div class="feature">
          <div class="feat-icon">🔌</div>
          <h3>OpenAI-Compatible API</h3>
          <p>Drop-in replacement for the Assistants API v2. Use the <code>openai</code> Python SDK without changing a single line of client code.</p>
        </div>
        <div class="feature">
          <div class="feat-icon">🤝</div>
          <h3>A2A Protocol</h3>
          <p>Built-in support for Google's Agent-to-Agent standard. Your agents are discoverable and interoperable with any A2A-compatible ecosystem.</p>
        </div>
        <div class="feature">
          <div class="feat-icon">🌊</div>
          <h3>Streaming &amp; Tool Calls</h3>
          <p>Real-time token streaming via SSE and native function calling. Works with OpenAI, Anthropic, Gemini, and any other LLM provider.</p>
        </div>
        <div class="feature">
          <div class="feat-icon">🔀</div>
          <h3>Multi-Agent Orchestration</h3>
          <p>Call or hand off to any agent in the same server with <code>call_agent()</code> and <code>handover_to_agent()</code> — no HTTP overhead.</p>
        </div>
        <div class="feature">
          <div class="feat-icon">📊</div>
          <h3>Compass Dashboard</h3>
          <p>Built-in observability UI for threads, runs, traces, and agent flow visualization. Live config store for runtime parameter tuning.</p>
        </div>
        <div class="feature">
          <div class="feat-icon">🐳</div>
          <h3>Production-Ready</h3>
          <p>Postgres, Redis, distributed workers, OpenTelemetry tracing, and Docker Compose — scale from a single file to a full cluster.</p>
        </div>
      </div>
    </div>

    <!-- ── Code ───────────────────────────────────── -->
    <div class="code-wrap">
      <div class="section-label">
        <h2>Simple by design</h2>
        <p>Write a plain Python class, configure in YAML, run with one command. No boilerplate.</p>
      </div>
      <div class="file-tab">agents.py</div>
      <pre><code><span class="kw">from</span> <span class="im">llamphouse.core</span> <span class="kw">import</span> Agent
<span class="kw">from</span> <span class="im">llamphouse.core.context</span> <span class="kw">import</span> Context


<span class="kw">class</span> <span class="cl">ChatAgent</span>(Agent):
    <span class="kw">async def</span> <span class="fn">run</span>(self, context: <span class="cl">Context</span>):
        <span class="cm"># context.messages       — full conversation history</span>
        <span class="cm"># context.send_chunk()   — stream tokens to the client in real-time</span>
        <span class="cm"># context.call_agent()   — delegate to another agent</span>
        <span class="cm"># context.get_config()   — read live runtime parameters</span>
        <span class="kw">await</span> context.insert_message(<span class="st">"Hello from LLAMPHouse! 🪔"</span>)</code></pre>
      <div class="file-tab" style="margin-top:1.25rem">llamphouse.yaml</div>
      <pre>version: "0.1"

definitions:
  - name: my-agent
    entrypoint: agents.py:ChatAgent

agents:
  - name: my-agent-prod
    definition: my-agent</pre>
      <div class="install" style="margin-top:1.25rem;display:inline-flex">
        <span class="install-prompt">$</span>
        <span class="install-cmd">llamphouse up</span>
      </div>
    </div>

    <!-- ── CTA ────────────────────────────────────── -->
    <div class="cta-wrap">
      <div class="cta">
        <h2>Ready to serve your first agent?</h2>
        <p>Install LLAMPHouse, follow the quickstart, and have a live agent endpoint in under 5 minutes.</p>
        <div class="hero-btns" style="margin-bottom: 0;">
          <a href="./docs/getting-started/quickstart/" class="btn-primary">
            Quickstart Guide
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </a>
          <a href="./docs/examples/" class="btn-outline">Browse Examples</a>
        </div>
      </div>
    </div>

  </main>

  <!-- ── Footer ─────────────────────────────────── -->
  <footer>
    <div class="footer-links">
      <a href="./docs/">Documentation</a>
      <a href="./docs/getting-started/quickstart/">Quickstart</a>
      <a href="./docs/examples/">Examples</a>
      <a href="./docs/concepts/adapters/">Adapters</a>
      <a href="./docs/contributing/">Contributing</a>
      <a href="https://github.com/llamp-ai/llamphouse" target="_blank" rel="noopener">GitHub</a>
    </div>
    <div>MIT License &nbsp;·&nbsp; LLAMPHouse</div>
  </footer>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# MkDocs hook
# ---------------------------------------------------------------------------
def on_post_build(config, **kwargs):
    """Write the landing page to ``site/index.html`` after the docs build."""
    site_dir = Path(config["site_dir"])        # .../site/docs
    landing_path = site_dir.parent / "index.html"  # .../site/index.html
    landing_path.parent.mkdir(parents=True, exist_ok=True)
    landing_path.write_text(LANDING_HTML, encoding="utf-8")
    print(f"  Landing page written → {landing_path}")
