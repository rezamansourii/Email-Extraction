#!/usr/bin/env python3
from __future__ import annotations

import io
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List

from flask import Flask, abort, render_template, request, send_file
from openpyxl import Workbook
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from email_extractor import CrawlConfig, crawl_and_extract

app = Flask(__name__)

RESULTS: Dict[str, dict] = {}


def _store_results(emails: List[str], visited: List[str], config: CrawlConfig) -> str:
    result_id = uuid.uuid4().hex
    RESULTS[result_id] = {
        "emails": emails,
        "visited": visited,
        "config": asdict(config),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return result_id


def _get_results(result_id: str) -> dict:
    data = RESULTS.get(result_id)
    if not data:
        abort(404)
    return data


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    result_id = None
    error = None

    if request.method == "POST":
        urls_raw = request.form.get("urls", "")
        urls = [line.strip() for line in urls_raw.splitlines() if line.strip()]

        if not urls:
            error = "Please provide at least one URL."
        else:
            cfg = CrawlConfig(
                depth=max(0, int(request.form.get("depth", 1))),
                max_pages=max(1, int(request.form.get("max_pages", 25))),
                delay_seconds=max(0.0, float(request.form.get("delay", 0.0))),
                timeout_seconds=max(1.0, float(request.form.get("timeout", 15.0))),
                user_agent=request.form.get(
                    "user_agent",
                    "EmailExtractor/1.0 (+https://example.local)",
                ),
            )
            do_crawl = request.form.get("crawl") == "on"

            emails, visited = crawl_and_extract(urls, cfg, do_crawl=do_crawl)
            emails_sorted = sorted(emails)
            visited_sorted = sorted(visited)
            result_id = _store_results(emails_sorted, visited_sorted, cfg)
            result = {
                "emails": emails_sorted,
                "visited": visited_sorted,
                "count": len(emails_sorted),
            }

    return render_template(
        "index.html",
        result=result,
        result_id=result_id,
        error=error,
    )


@app.route("/export/excel/<result_id>")
def export_excel(result_id: str):
    data = _get_results(result_id)

    wb = Workbook()
    ws = wb.active
    ws.title = "Emails"
    ws.append(["Email"])
    for email in data["emails"]:
        ws.append([email])

    ws_meta = wb.create_sheet(title="Metadata")
    ws_meta.append(["Generated At", data["created_at"]])
    ws_meta.append(["Total Emails", len(data["emails"])])
    ws_meta.append(["Visited Pages", len(data["visited"])])
    ws_meta.append(["Config", str(data["config"])])

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    filename = f"emails-{result_id}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/export/pdf/<result_id>")
def export_pdf(result_id: str):
    data = _get_results(result_id)
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin = 0.75 * inch
    y = height - margin
    line_height = 14

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, y, "Email Extraction Report")
    y -= line_height * 1.5

    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin, y, f"Generated: {data['created_at']}")
    y -= line_height
    pdf.drawString(margin, y, f"Total Emails: {len(data['emails'])}")
    y -= line_height
    pdf.drawString(margin, y, f"Visited Pages: {len(data['visited'])}")
    y -= line_height * 1.5

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Emails")
    y -= line_height

    pdf.setFont("Helvetica", 10)
    for email in data["emails"]:
        if y <= margin:
            pdf.showPage()
            y = height - margin
            pdf.setFont("Helvetica", 10)
        pdf.drawString(margin, y, email)
        y -= line_height

    pdf.save()
    buffer.seek(0)

    filename = f"emails-{result_id}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
