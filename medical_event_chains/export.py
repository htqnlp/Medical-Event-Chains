import json
from pathlib import Path


def standalone_html(result):
    web = Path(__file__).parent / "web"
    template = (web / "index.html").read_text(encoding="utf-8")
    css = (web / "style.css").read_text(encoding="utf-8")
    script = (web / "app.js").read_text(encoding="utf-8")
    # Prevent source text containing </script> from escaping the JSON script element.
    data = json.dumps(result, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace('<link rel="stylesheet" href="/style.css">', f"<style>{css}</style>").replace('<script src="/app.js" defer></script>', f'<script type="application/json" id="mec-data">{data}</script><script defer>{script}</script>')


def write_report(result, output):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    json_path, html_path = destination / "report.json", destination / "report.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    html_path.write_text(standalone_html(result), encoding="utf-8")
    return json_path, html_path
