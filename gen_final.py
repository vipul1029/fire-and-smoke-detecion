


# gen_final.py — Final polish of the v2 presentation.
# Same 8-slide content; improved typography, spacing, and methodology diagram.

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── colour palette ─────────────────────────────────────────────────────────────
N1  = RGBColor(0x1A, 0x28, 0x5C)   # Navy primary
N2  = RGBColor(0x12, 0x1C, 0x42)   # Navy deep
N3  = RGBColor(0x2B, 0x3A, 0x7A)   # Navy mid
RE  = RGBColor(0xBE, 0x22, 0x22)   # Fire red
OR  = RGBColor(0xDA, 0x58, 0x00)   # Smoke orange
TE  = RGBColor(0x00, 0x7A, 0x8A)   # Teal
GR  = RGBColor(0x15, 0x7A, 0x3C)   # Green
PU  = RGBColor(0x5B, 0x2D, 0x8E)   # Purple
BL  = RGBColor(0x15, 0x5D, 0xAF)   # Blue
WH  = RGBColor(0xFF, 0xFF, 0xFF)   # White
LG  = RGBColor(0xF4, 0xF6, 0xFB)   # Light grey bg
MG  = RGBColor(0xE2, 0xE6, 0xF0)   # Mid grey
DG  = RGBColor(0x55, 0x65, 0x7A)   # Dark grey (muted text)
TXT = RGBColor(0x1A, 0x20, 0x30)   # Near-black body text
GD  = RGBColor(0xF0, 0x8C, 0x00)   # Gold accent

# ── slide geometry ──────────────────────────────────────────────────────────────
SW  = Inches(13.33)
SH  = Inches(7.50)
HH  = Inches(1.05)          # header band height
FH  = Inches(0.30)          # footer bar height
LM  = Inches(0.42)          # left margin
RM  = Inches(0.42)          # right margin (from right edge)
CW  = SW - LM - Inches(0.42)  # full content width ≈ 12.49"
CT  = HH + Inches(0.18)     # content top
CB  = SH - FH - Inches(0.12) # content bottom
CH  = CB - CT               # ≈ 5.85"

def rgb(r, g, b):
    return RGBColor(r, g, b)

# ── primitive helpers ───────────────────────────────────────────────────────────
def box(sl, l, t, w, h, fill=None, border=None, bw=None, radius=None):
    shp = sl.shapes.add_shape(1, l, t, w, h)  # MSO_SHAPE_TYPE.RECTANGLE
    shp.line.width = Pt(bw) if bw else Pt(0)
    if border:
        shp.line.color.rgb = border
    else:
        shp.line.fill.background()
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    else:
        shp.fill.background()
    return shp

def rbox(sl, l, t, w, h, fill=None, border=None, bw=None, corner=Pt(6)):
    """Rounded rectangle."""
    from pptx.util import Emu
    shp = sl.shapes.add_shape(5, l, t, w, h)  # ROUNDED_RECTANGLE
    shp.adjustments[0] = 0.05
    shp.line.width = Pt(bw) if bw else Pt(0)
    if border:
        shp.line.color.rgb = border
    else:
        shp.line.fill.background()
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    else:
        shp.fill.background()
    return shp

def txt(sl, text, l, t, w, h, size=9, bold=False, color=TXT,
        align=PP_ALIGN.LEFT, wrap=True, italic=False, spacing=None):
    tb = sl.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    if spacing:
        p.line_spacing = spacing
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb

def mtxt(sl, lines, l, t, w, h, size=9, bold=False, color=TXT,
         align=PP_ALIGN.LEFT, spacing=1.15):
    """Multi-paragraph textbox — list of (text, bold_override) or plain strings."""
    from pptx.util import Pt as _Pt
    tb = sl.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in lines:
        if isinstance(item, tuple):
            line_text, lb = item
        else:
            line_text, lb = item, bold
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        run = p.add_run()
        run.text = line_text
        run.font.size = _Pt(size)
        run.font.bold = lb
        run.font.color.rgb = color
    return tb

def arrow_down(sl, cx, top, height=Inches(0.16), color=MG, w=Pt(1.5)):
    """Thin downward connector arrow."""
    from pptx.util import Emu
    connector = sl.shapes.add_connector(1, cx, top, cx, top + height)
    connector.line.color.rgb = color
    connector.line.width = w
    return connector

def hline(sl, l, t, w, color=MG, lw=Pt(0.5)):
    ln = sl.shapes.add_shape(1, l, t, w, Inches(0))
    ln.line.color.rgb = color
    ln.line.width = lw
    ln.fill.background()
    return ln

def badge(sl, text, l, t, w, h, fill, size=7.5, bold=True, color=WH):
    b = box(sl, l, t, w, h, fill=fill)
    txt(sl, text, l, t, w, h, size=size, bold=bold, color=color,
        align=PP_ALIGN.CENTER)
    return b

def new_prs():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs

# ── shared header / footer ──────────────────────────────────────────────────────
def header(sl, title_text, slide_num):
    # Deep navy gradient header band
    box(sl, Inches(0), Inches(0), SW, HH, fill=N2)
    # Left accent stripe
    box(sl, Inches(0), Inches(0), Inches(0.06), HH, fill=RE)
    # Slide title (centred)
    txt(sl, title_text,
        Inches(0.12), Inches(0.12),
        SW - Inches(0.24), Inches(0.75),
        size=18, bold=True, color=WH, align=PP_ALIGN.CENTER)
    # Bottom highlight line on header
    hline(sl, Inches(0), HH - Inches(0.025), SW, color=RE, lw=Pt(2))

def footer(sl, slide_num):
    FT = SH - FH
    box(sl, Inches(0), FT, SW, FH, fill=N2)
    txt(sl, "Vellore Institute of Technology  |  Real-Time Fire & Smoke Detection  |  Final Year Project",
        Inches(0.3), FT + Inches(0.05), Inches(9.0), FH,
        size=7, color=RGBColor(0xAA, 0xB4, 0xCC), align=PP_ALIGN.LEFT)
    badge(sl, f"{slide_num} / 8",
          SW - Inches(0.9), FT + Inches(0.04),
          Inches(0.7), FH - Inches(0.08),
          fill=N1, size=7.5, bold=True, color=WH)

# ── SLIDE 1: TITLE ─────────────────────────────────────────────────────────────
def s1_title(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])

    # Full background
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)

    # Left hero band (navy)
    LW = Inches(8.20)
    box(sl, Inches(0), Inches(0), LW, SH, fill=N1)

    # Left red accent stripe
    box(sl, Inches(0), Inches(0), Inches(0.07), SH, fill=RE)

    # Decorative diagonal stripe pattern (subtle)
    for i in range(6):
        box(sl, Inches(0.07) + Inches(0.55*i), Inches(0),
            Inches(0.25), SH, fill=N2)

    # Institution badge
    badge(sl, "VIT  |  Vellore Institute of Technology",
          Inches(0.3), Inches(0.30),
          Inches(5.5), Inches(0.35),
          fill=RE, size=8.5, bold=True, color=WH)

    # Subtitle tag
    txt(sl, "Final Year Project  •  Computer Science & Engineering  •  AI/ML",
        Inches(0.3), Inches(0.75),
        Inches(7.2), Inches(0.35),
        size=8.5, color=RGBColor(0xAA, 0xBF, 0xDD), bold=False)

    # Main title
    tb = sl.shapes.add_textbox(Inches(0.30), Inches(1.22), Inches(7.7), Inches(1.65))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = "Real-Time Fire and\nSmoke Detection"
    r.font.size = Pt(38); r.font.bold = True; r.font.color.rgb = WH

    # Subtitle
    txt(sl, "YOLOv8-Based Computer Vision System for Intelligent Hazard Monitoring",
        Inches(0.30), Inches(2.90),
        Inches(7.5), Inches(0.55),
        size=12.5, color=RGBColor(0xCC, 0xD8, 0xEE), italic=True)

    # Divider
    hline(sl, Inches(0.30), Inches(3.58), Inches(6.80), color=RE, lw=Pt(1.8))

    # Team info block
    txt(sl, "Presented by",
        Inches(0.30), Inches(3.70), Inches(3.5), Inches(0.30),
        size=8, color=RGBColor(0xAA, 0xBF, 0xDD), bold=False)

    team_lines = [
        "[Team Member 1]  •  Reg. No. [XXXXXXXXXX]",
        "[Team Member 2]  •  Reg. No. [XXXXXXXXXX]",
    ]
    mtxt(sl, team_lines,
         Inches(0.30), Inches(4.00), Inches(5.0), Inches(0.65),
         size=9, bold=True, color=WH)

    txt(sl, "Project Guide:  [Faculty Name],  [Department]",
        Inches(0.30), Inches(4.70), Inches(5.5), Inches(0.35),
        size=8.5, color=RGBColor(0xCC, 0xD8, 0xEE))

    txt(sl, "Academic Year: 2025–2026",
        Inches(0.30), Inches(5.08), Inches(4.5), Inches(0.30),
        size=8, color=RGBColor(0xAA, 0xBF, 0xDD))

    # ── Right panel: system pipeline concept ──────────────────────────
    RP = Inches(8.42)
    RW = SW - RP - Inches(0.22)

    txt(sl, "SYSTEM PIPELINE OVERVIEW",
        RP, Inches(0.35), RW, Inches(0.32),
        size=8, bold=True, color=N1, align=PP_ALIGN.CENTER)

    pipeline_steps = [
        (OR,   "\U0001f4f7  IMAGE / VIDEO / WEBCAM",   "Input sources"),
        (N1,   "\U0001f50d  YOLOv8-BASED DETECTION",   "Fire & smoke localisation"),
        (RE,   "\U0001f4e6  BOUNDING BOXES + CONF",    "Visual output with confidence"),
        (GR,   "\U0001f4ca  SEVERITY + FPS MONITOR",   "Real-time hazard scoring"),
    ]

    pbox_h  = Inches(0.88)
    pbox_gap = Inches(0.16)
    ptop = Inches(0.72)

    for i, (col, label, sub) in enumerate(pipeline_steps):
        bt = ptop + i * (pbox_h + pbox_gap)
        rbox(sl, RP, bt, RW, pbox_h, fill=col)
        txt(sl, label,
            RP + Inches(0.10), bt + Inches(0.10),
            RW - Inches(0.20), Inches(0.38),
            size=9, bold=True, color=WH, align=PP_ALIGN.CENTER)
        txt(sl, sub,
            RP + Inches(0.10), bt + Inches(0.48),
            RW - Inches(0.20), Inches(0.32),
            size=7.5, color=WH, align=PP_ALIGN.CENTER)
        # Arrow (not after last)
        if i < len(pipeline_steps) - 1:
            cx = RP + RW / 2
            at = bt + pbox_h
            arrow_down(sl, cx, at, pbox_gap - Inches(0.02),
                       color=DG, w=Pt(1.2))

    # Keyword strip at bottom of right panel
    kw_top = Inches(5.30)
    box(sl, RP, kw_top, RW, Inches(0.35), fill=N2)
    txt(sl, "Python  •  OpenCV  •  Ultralytics  •  YOLOv8",
        RP + Inches(0.06), kw_top + Inches(0.06),
        RW - Inches(0.12), Inches(0.24),
        size=7.5, color=RGBColor(0xAA, 0xBF, 0xDD), align=PP_ALIGN.CENTER, bold=True)

    # ── global footer on title ──────────────────────────────────────────
    FT = SH - Inches(0.32)
    box(sl, Inches(0), FT, SW, Inches(0.32), fill=N2)
    txt(sl, "Academic Research Presentation  —  Not a product deployment",
        Inches(0.3), FT + Inches(0.05), SW - Inches(0.6), Inches(0.22),
        size=7, color=RGBColor(0xAA, 0xB4, 0xCC), align=PP_ALIGN.CENTER)


# ── SLIDE 2: PROBLEM MOTIVATION ────────────────────────────────────────────────
def s2_problem(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)
    header(sl, "PROBLEM MOTIVATION", 2)
    footer(sl, 2)

    # Section label
    txt(sl, "Why automated real-time fire and smoke detection is a critical research challenge",
        LM, HH + Inches(0.14), CW, Inches(0.28),
        size=9, color=DG, italic=True, align=PP_ALIGN.CENTER)

    # 4 challenge cards (2 × 2)
    cards = [
        (RE,   "01",  "Rapid Escalation",
                "Uncontrolled fire can spread from ignition to full room-engulfment in under 3 minutes "
                "(NFPA statistics). Human-visible smoke precedes dangerous flame by 30–90 seconds, "
                "making early automated detection essential."),
        (OR,   "02",  "Visual Complexity",
                "Smoke and fire share visual features with steam, sunset glare, and dust clouds. "
                "Illumination variance, partial occlusion, and varying colour temperatures cause "
                "persistent false-positive and missed-detection failures in rule-based systems."),
        (N1,   "03",  "Limits of Manual Monitoring",
                "CCTV operators suffer attention degradation above 20-minute monitoring windows "
                "(HSE studies). Large-campus deployments with hundreds of cameras make continuous "
                "manual vigilance infeasible, creating temporal blind spots."),
        (TE,   "04",  "Accuracy–Efficiency Trade-off",
                "Two-stage detectors (Faster R-CNN, Cascade R-CNN) achieve higher precision but "
                "cannot sustain real-time throughput on commodity hardware. Lightweight models "
                "sacrifice recall; no single published approach resolves this tension fully."),
    ]

    card_w = (CW - Inches(0.28)) / 2
    card_h = Inches(2.18)
    card_rows = 2
    card_cols = 2

    for idx, (col, num, title, body) in enumerate(cards):
        row = idx // 2
        col_i = idx % 2
        cl = LM + col_i * (card_w + Inches(0.28))
        ct_ = CT + Inches(0.38) + row * (card_h + Inches(0.20))

        rbox(sl, cl, ct_, card_w, card_h, fill=WH, border=col, bw=1.8)
        # Top accent bar
        box(sl, cl, ct_, card_w, Inches(0.06), fill=col)
        # Number badge
        badge(sl, num, cl + Inches(0.14), ct_ + Inches(0.12),
              Inches(0.38), Inches(0.32), fill=col, size=10, bold=True, color=WH)
        # Title
        txt(sl, title,
            cl + Inches(0.60), ct_ + Inches(0.14),
            card_w - Inches(0.72), Inches(0.35),
            size=11, bold=True, color=col)
        # Body
        txt(sl, body,
            cl + Inches(0.14), ct_ + Inches(0.56),
            card_w - Inches(0.28), Inches(1.52),
            size=8.5, color=TXT)

    # Bottom flow strip
    strip_top = CT + Inches(0.38) + 2 * (card_h + Inches(0.20))
    box(sl, LM, strip_top, CW, Inches(0.58), fill=N1)
    flow_items = [
        ("DETECTION GAP",     RE,   "Manual systems miss sub-minute ignition windows"),
        ("→",             None, None),
        ("RESEARCH NEED",     OR,   "Automated, real-time, multi-class visual detection"),
        ("→",             None, None),
        ("PROJECT RESPONSE",  GR,   "YOLOv8-based inference pipeline (this project)"),
    ]
    fw = CW / len(flow_items)
    for i, (label, col, sub) in enumerate(flow_items):
        fl = LM + i * fw
        if col is None:
            txt(sl, label,
                fl + fw*0.3, strip_top + Inches(0.15),
                fw * 0.4, Inches(0.30),
                size=14, bold=True, color=WH, align=PP_ALIGN.CENTER)
        else:
            txt(sl, label,
                fl + Inches(0.06), strip_top + Inches(0.06),
                fw - Inches(0.12), Inches(0.26),
                size=7.5, bold=True, color=col, align=PP_ALIGN.CENTER)
            txt(sl, sub,
                fl + Inches(0.06), strip_top + Inches(0.30),
                fw - Inches(0.12), Inches(0.24),
                size=7, color=RGBColor(0xCC, 0xD8, 0xEE), align=PP_ALIGN.CENTER)


# ── SLIDE 3: LITERATURE REVIEW ─────────────────────────────────────────────────
def s3_literature(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)
    header(sl, "LITERATURE REVIEW: WHAT EXISTING RESEARCH SHOWS", 3)
    footer(sl, 3)

    # Summary banner
    box(sl, LM, CT + Inches(0.12), CW, Inches(0.38), fill=N1)
    txt(sl, "15 Research Papers Reviewed  •  Grouped into 4 Thematic Areas  "
            "•  Key findings distilled per theme",
        LM + Inches(0.18), CT + Inches(0.22),
        CW - Inches(0.36), Inches(0.24),
        size=9, bold=True, color=WH, align=PP_ALIGN.CENTER)

    themes = [
        ("T1", BL,  "CNN-Based Fire Detection\n(5 papers)",
         [("Frizzi et al. (2016):", "Early CNN baseline; 85–90% accuracy on controlled datasets."),
          ("Muhammad et al. (2018):", "AlexNet/GoogleNet fusion; improved recall but high compute."),
          ("Chino et al. (2015):", "BoWFire: bag-of-words + SVM; illumination-sensitive."),
          ("Dunnings & Breckon (2018):", "InceptionV1 classifier; strong on static cameras."),
          ("Sharma et al. (2023):", "EfficientNet transfer learning; compact and fast.")]),
        ("T2", RE,  "YOLO-Family Detectors\n(4 papers)",
         [("Luo et al. (2022):", "YOLOv5s fine-tuned for fire; 72.3% mAP@0.5."),
          ("Li et al. (2023):", "YOLOv7-tiny + attention; balanced accuracy–speed."),
          ("Zhang & Li (2023):", "YOLO-FSD with feature pyramid; improved small-smoke recall."),
          ("Wan et al. (2023):", "YOLOv8 fire baseline; confirms suitability for real-time deployment.")]),
        ("T3", OR,  "Transformers & Attention\n(3 papers)",
         [("Bochkovskiy et al. (2020):", "CSPNet backbone reduces parameters, preserves gradient flow."),
          ("Khan et al. (2022):", "ViT self-attention; high precision, ∼3× slower than YOLO."),
          ("Park et al. (2023):", "Hybrid CNN-transformer; accuracy gains not sustained at video frame rates.")]),
        ("T4", GR,  "Severity & Alert Systems\n(3 papers)",
         [("Toulouse et al. (2017):", "Pixel-ratio severity index; simple but scene-dependent."),
          ("Filonenko et al. (2016):", "Optical-flow smoke; temporal context improves true positives."),
          ("Foggia et al. (2015):", "SVM + motion; performance degrades in low-frame-rate feeds.")]),
    ]

    card_w = (CW - Inches(0.24)) / 2
    card_h = Inches(2.66)

    for idx, (tag, col, title, refs) in enumerate(themes):
        row = idx // 2
        ci  = idx % 2
        cl  = LM + ci * (card_w + Inches(0.24))
        ct_ = CT + Inches(0.62) + row * (card_h + Inches(0.18))

        rbox(sl, cl, ct_, card_w, card_h, fill=WH, border=col, bw=1.5)
        box(sl, cl, ct_, card_w, Inches(0.06), fill=col)

        # Tag + title on same header bar
        box(sl, cl, ct_ + Inches(0.06), card_w, Inches(0.48), fill=col)
        badge(sl, tag, cl + Inches(0.08), ct_ + Inches(0.10),
              Inches(0.34), Inches(0.30), fill=N2, size=9, bold=True, color=WH)
        # Title lines (two lines)
        title_parts = title.split("\n")
        txt(sl, title_parts[0],
            cl + Inches(0.50), ct_ + Inches(0.10),
            card_w - Inches(0.62), Inches(0.24),
            size=9, bold=True, color=WH)
        if len(title_parts) > 1:
            txt(sl, title_parts[1],
                cl + Inches(0.50), ct_ + Inches(0.32),
                card_w - Inches(0.62), Inches(0.18),
                size=7.5, color=RGBColor(0xEE, 0xF2, 0xFF))

        # Reference entries
        ey = ct_ + Inches(0.62)
        for author, finding in refs:
            # Author
            txt(sl, author,
                cl + Inches(0.12), ey,
                card_w - Inches(0.24), Inches(0.22),
                size=7.5, bold=True, color=col)
            ey += Inches(0.22)
            txt(sl, finding,
                cl + Inches(0.12), ey,
                card_w - Inches(0.24), Inches(0.24),
                size=7.5, color=TXT)
            ey += Inches(0.26)


# ── SLIDE 4: RESEARCH GAP ──────────────────────────────────────────────────────
def s4_gap(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)
    header(sl, "RESEARCH GAP", 4)
    footer(sl, 4)

    txt(sl, "Critical deficiencies identified across the 15-paper survey",
        LM, CT + Inches(0.14), CW, Inches(0.28),
        size=9, color=DG, italic=True, align=PP_ALIGN.CENTER)

    gaps = [
        (RE,  "G1",  "Accuracy–Speed Trade-off",
               "Many architectures improve detection accuracy through costly custom backbones, "
               "precluding real-time (>25 fps) inference on embedded or commodity hardware."),
        (OR,  "G2",  "Robustness Under Adverse Conditions",
               "Performance degrades significantly under low-light, dense-smoke occlusion, "
               "and foggy backgrounds. No published solution sustains >70% recall across all such conditions."),
        (N1,  "G3",  "Lack of Unified Pipelines",
               "Most works address detection only. Real deployments require input handling, "
               "bounding-box annotation, severity estimation, and monitoring in a single integrated system."),
        (PU,  "G4",  "Severity Quantification Absent",
               "Fewer than 3 of 15 reviewed papers propose any severity or urgency score. "
               "None combine fire count, smoke count, and frame rate into a deployable heuristic."),
        (TE,  "G5",  "Limited Multi-Class Evaluation",
               "Most datasets label fire only. Simultaneous fire–smoke multi-class detection "
               "with class-specific recall metrics remains under-studied in real-time settings."),
    ]

    gap_h  = Inches(0.90)
    gap_gap = Inches(0.13)
    GW = CW * 0.62

    for i, (col, num, title, body) in enumerate(gaps):
        gt = CT + Inches(0.46) + i * (gap_h + gap_gap)
        rbox(sl, LM, gt, GW, gap_h, fill=WH, border=col, bw=1.5)
        box(sl, LM, gt, Inches(0.06), gap_h, fill=col)
        badge(sl, num, LM + Inches(0.12), gt + Inches(0.27),
              Inches(0.34), Inches(0.30), fill=col, size=8, bold=True, color=WH)
        txt(sl, title,
            LM + Inches(0.56), gt + Inches(0.10),
            GW - Inches(0.68), Inches(0.30),
            size=10, bold=True, color=col)
        txt(sl, body,
            LM + Inches(0.56), gt + Inches(0.40),
            GW - Inches(0.68), Inches(0.46),
            size=8.5, color=TXT)

    # Right summary panel
    RP = LM + GW + Inches(0.30)
    RW = CW - GW - Inches(0.30)
    RT = CT + Inches(0.46)
    RH = 5 * gap_h + 4 * gap_gap

    rbox(sl, RP, RT, RW, RH, fill=N1)
    txt(sl, "THIS PROJECT\nADDRESSES",
        RP + Inches(0.14), RT + Inches(0.20),
        RW - Inches(0.28), Inches(0.60),
        size=10.5, bold=True, color=WH, align=PP_ALIGN.CENTER)

    hline(sl, RP + Inches(0.20), RT + Inches(0.88), RW - Inches(0.40), color=RE, lw=Pt(1.5))

    responses = [
        (RE,  "G1", "YOLOv8 one-stage\narchitecture for real-time"),
        (OR,  "G2", "Benchmark on D-Fire\nincl. adverse conditions"),
        (N1,  "G3", "Integrated end-to-end\ndetection pipeline"),
        (PU,  "G4", "Severity heuristic:\nS = min(100, F×45+Sm×20+FPS×2)"),
        (TE,  "G5", "Dual-class inference:\nfire(1) + smoke(0)"),
    ]

    ry = RT + Inches(1.00)
    rh = (RH - Inches(1.20)) / len(responses)
    for col, num, resp in responses:
        badge(sl, num, RP + Inches(0.14), ry + Inches(0.06),
              Inches(0.34), Inches(0.26), fill=col, size=7.5, bold=True)
        txt(sl, resp,
            RP + Inches(0.55), ry + Inches(0.04),
            RW - Inches(0.68), rh - Inches(0.08),
            size=7.5, color=WH)
        ry += rh


# ── SLIDE 5: RESEARCH OBJECTIVES ──────────────────────────────────────────────
def s5_objectives(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)
    header(sl, "RESEARCH OBJECTIVES", 5)
    footer(sl, 5)

    objectives = [
        (N1, "01", "Real-Time Fire and Smoke Detection Pipeline",
               "Develop a unified inference pipeline that detects and localises fire and smoke "
               "from image, video, and live webcam sources using a pre-trained YOLOv8-based model; "
               "sustain processing at interactive frame rates on standard hardware.",
               "IMPLEMENTED"),
        (RE, "02", "Confidence-Filtered Multi-Class Detection",
               "Apply class-specific confidence thresholding (τₜᵒⁿf = 0.25) and IoU-based "
               "Non-Maximum Suppression (τ_IoU = 0.45) to separate fire (class 1) and smoke (class 0) "
               "detections, suppress redundant overlapping boxes, and output precise bounding box coordinates.",
               "IMPLEMENTED"),
        (OR, "03", "Project-Specific Severity Heuristic",
               "Define and implement a severity scoring function S = min(100, F×45 + Sm×20 + FPS×2) "
               "that combines fire count (F), smoke count (Sm), and rolling-window FPS into a "
               "four-tier urgency label (critical / high / medium / low) displayed in real time.",
               "IMPLEMENTED"),
        (TE, "04", "Structured Quantitative Evaluation",
               "Design a rigorous evaluation framework using standard metrics (Precision, Recall, F1-Score, "
               "mAP@0.5, mAP@0.5:0.95) and assess performance on D-Fire as a potential benchmark dataset. "
               "Benchmark results will be reported after independent experimental evaluation.",
               "PLANNED"),
        (PU, "05", "Comparative Benchmarking and Ablation Study",
               "Compare YOLOv8-based detection against established baselines (YOLOv5, Faster R-CNN) "
               "on fire and smoke benchmarks; conduct ablation studies on confidence threshold, "
               "input resolution, and NMS parameters to characterise performance trade-offs.",
               "PLANNED"),
    ]

    obj_h  = Inches(0.96)
    obj_gap = Inches(0.12)
    NW = Inches(0.55)   # number badge width
    SW_ = Inches(1.12)  # status badge width

    for i, (col, num, title, body, status) in enumerate(objectives):
        ot = CT + Inches(0.22) + i * (obj_h + obj_gap)
        rbox(sl, LM, ot, CW, obj_h, fill=WH, border=col, bw=1.5)
        box(sl, LM, ot, Inches(0.06), obj_h, fill=col)
        # Number
        badge(sl, num, LM + Inches(0.10), ot + Inches(0.30),
              NW, Inches(0.34), fill=col, size=11, bold=True)
        # Status badge
        scol = GR if status == "IMPLEMENTED" else OR
        badge(sl, status,
              LM + CW - SW_ - Inches(0.12), ot + Inches(0.32),
              SW_, Inches(0.30), fill=scol, size=7.5, bold=True)
        # Title
        txt(sl, title,
            LM + NW + Inches(0.24), ot + Inches(0.10),
            CW - NW - SW_ - Inches(0.48), Inches(0.30),
            size=10.5, bold=True, color=col)
        # Body
        txt(sl, body,
            LM + NW + Inches(0.24), ot + Inches(0.42),
            CW - NW - SW_ - Inches(0.48), Inches(0.50),
            size=8.5, color=TXT)


# ── SLIDE 6: METHODOLOGY ──────────────────────────────────────────────────────
def s6_methodology(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)
    header(sl, "PROPOSED METHODOLOGY", 6)
    footer(sl, 6)

    # ── Layout geometry ────────────────────────────────────────────────────
    LABEL_W  = Inches(1.52)    # phase label column
    SPEC_W   = Inches(2.18)    # right annotation column
    GAP      = Inches(0.10)    # gap between columns
    MAIN_W   = CW - LABEL_W - SPEC_W - GAP * 2
    MAIN_L   = LM + LABEL_W + GAP
    SPEC_L   = MAIN_L + MAIN_W + GAP

    # Column headers
    for (l, w, lbl) in [
        (LM,     LABEL_W, "PHASE"),
        (MAIN_L, MAIN_W,  "PIPELINE STAGE — OPERATION"),
        (SPEC_L, SPEC_W,  "TECHNICAL SPEC / SOURCE"),
    ]:
        box(sl, l, CT + Inches(0.10), w, Inches(0.32), fill=N1)
        txt(sl, lbl,
            l + Inches(0.06), CT + Inches(0.14),
            w - Inches(0.12), Inches(0.24),
            size=7.5, bold=True, color=WH, align=PP_ALIGN.CENTER)

    # ── Pipeline steps ─────────────────────────────────────────────────────
    steps = [
        # (phase_color, phase_label, step_title, step_body, spec_text)
        (TE,  "INPUT\nACQUISITION",
               "Input Acquisition",
               "Image: .jpg / .png     •     Video: .mp4 / .avi / .mov / .mkv     •     Live Webcam stream",
               "run.py → cv2.imread\ncv2.VideoCapture(0)"),

        (BL,  "FRAME PRE-\nPROCESSING",
               "Frame Preprocessing  [Ultralytics internal — not custom code]",
               "Letterbox resize to 640×640 px     |     Pixel normalisation ÷255 → [0, 1]     |     BGR→RGB     |     NumPy array → PyTorch tensor",
               "imgsz = 640\nUltralytics YOLO()"),

        (N1,  "YOLOv8\nINFERENCE",
               "YOLOv8-Based Model Inference",
               "Backbone: CSP-Darknet + C2f residual modules     •     Neck: PANet-FPN (multi-scale feature aggregation)     •     Head: Decoupled anchor-free prediction",
               "fire_detection_yolov8m.pt\n(checkpoint: YOLOv8n-class)"),

        (OR,  "CONFIDENCE\nFILTERING",
               "Confidence Filtering",
               "Retain detection d if  conf(d) ≥ τₜᵒⁿf = 0.25     •     Class 0 = smoke     •     Class 1 = fire     •     Output: (class, conf, bboxₜₜₜ)",
               "τₜᵒⁿf = 0.25\nsrc/detector.py:57\n(configurable via --conf)"),

        (RE,  "NON-MAXIMUM\nSUPPRESSION",
               "Non-Maximum Suppression",
               "For overlapping boxes:  IoU(A, B) = |A∩B| / |A∪B|     •     If IoU > τ_IoU = 0.45 → remove lower-confidence box     •     Retain highest-conf per object",
               "τ_IoU = 0.45\nsrc/detector.py:58"),

        (PU,  "SEVERITY\nSCORING",
               "Severity Score + FPS Monitoring",
               "S = min(100, F×45 + Sm×20 + FPS×2)   [project-specific heuristic]     •     Labels: critical(≥70) / high(≥40) / medium(≥15) / low     •     FPS: 30-frame rolling window",
               "src/pipeline.py:98\nFPS window: 30 frames\nF = fire count, Sm = smoke"),

        (GR,  "ANNOTATED\nOUTPUT",
               "Annotated Detection Output",
               "Red bbox = fire     •     Orange bbox = smoke     •     Confidence label per box     •     HUD overlay: FPS, fire/smoke counts     •     Write to file / display / stream",
               "src/pipeline.py:75\ncv2.imwrite / VideoWriter\ncv2.imshow"),
    ]

    STEP_H  = Inches(0.66)
    ARR_H   = Inches(0.13)
    total_h = len(steps) * STEP_H + (len(steps) - 1) * ARR_H
    START_Y = CT + Inches(0.52)

    for i, (col, phase, title, body, spec) in enumerate(steps):
        sy = START_Y + i * (STEP_H + ARR_H)

        # Phase label block (left)
        box(sl, LM, sy, LABEL_W, STEP_H, fill=col)
        txt(sl, phase,
            LM + Inches(0.06), sy + Inches(0.05),
            LABEL_W - Inches(0.12), STEP_H - Inches(0.10),
            size=7.5, bold=True, color=WH, align=PP_ALIGN.CENTER)

        # Main content block
        rbox(sl, MAIN_L, sy, MAIN_W, STEP_H, fill=WH, border=col, bw=0.8)
        # Step title
        txt(sl, title,
            MAIN_L + Inches(0.10), sy + Inches(0.07),
            MAIN_W - Inches(0.20), Inches(0.26),
            size=9, bold=True, color=col)
        # Step body
        txt(sl, body,
            MAIN_L + Inches(0.10), sy + Inches(0.34),
            MAIN_W - Inches(0.20), Inches(0.28),
            size=7.5, color=TXT)

        # Spec annotation (right)
        box(sl, SPEC_L, sy, SPEC_W, STEP_H,
            fill=RGBColor(0xF0, 0xF4, 0xFA), border=MG, bw=0.6)
        txt(sl, spec,
            SPEC_L + Inches(0.08), sy + Inches(0.08),
            SPEC_W - Inches(0.16), STEP_H - Inches(0.16),
            size=7.5, color=DG)

        # Arrow connector (not after last)
        if i < len(steps) - 1:
            arrow_down(sl, LM + LABEL_W / 2, sy + STEP_H,
                       ARR_H, color=col, w=Pt(1.0))
            arrow_down(sl, MAIN_L + MAIN_W / 2, sy + STEP_H,
                       ARR_H, color=col, w=Pt(1.0))


# ── SLIDE 7: EVALUATION FRAMEWORK ─────────────────────────────────────────────
def s7_evaluation(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)
    header(sl, "EVALUATION FRAMEWORK", 7)
    footer(sl, 7)

    # ── Left column: metric cards ────────────────────────────────────────
    MC_W = Inches(3.80)
    MC_L = LM
    txt(sl, "PROPOSED EVALUATION METRICS",
        MC_L, CT + Inches(0.14), MC_W, Inches(0.28),
        size=8.5, bold=True, color=N1, align=PP_ALIGN.CENTER)

    metrics = [
        (BL,  "Precision",       "TP / (TP + FP)",
               "Fraction of detected objects that are true fire/smoke events.",
               "class-wise: fire & smoke"),
        (GR,  "Recall",          "TP / (TP + FN)",
               "Fraction of actual events that the model successfully detects.",
               "class-wise: fire & smoke"),
        (OR,  "F1-Score",        "2 × (P × R) / (P + R)",
               "Harmonic mean of Precision and Recall; balances both measures.",
               "primary ranking metric"),
        (N1,  "mAP@0.5",         "∑ APᵏ / C   (IoU ≥ 0.5)",
               "Mean average precision at IoU threshold 0.50 across both classes.",
               "standard detection benchmark"),
        (RE,  "mAP@0.5:0.95",    "∑ APᵏ / (10×C)   (IoU 0.5–0.95)",
               "COCO-style mean AP averaged over 10 IoU thresholds; stricter metric.",
               "strict localisation quality"),
        (PU,  "Inference FPS",   "frames / elapsed (30-frame window)",
               "Rolling 30-frame throughput; target: interactive (≥15 fps) frame rate.",
               "src/pipeline.py:46–51"),
    ]

    mc_h = Inches(0.70)
    mc_gap = Inches(0.10)
    for i, (col, name, formula, desc, note) in enumerate(metrics):
        mt = CT + Inches(0.50) + i * (mc_h + mc_gap)
        rbox(sl, MC_L, mt, MC_W, mc_h, fill=WH, border=col, bw=1.4)
        box(sl, MC_L, mt, Inches(0.06), mc_h, fill=col)
        txt(sl, name,
            MC_L + Inches(0.12), mt + Inches(0.07),
            Inches(1.50), Inches(0.26),
            size=9.5, bold=True, color=col)
        txt(sl, formula,
            MC_L + Inches(1.68), mt + Inches(0.07),
            MC_W - Inches(1.80), Inches(0.26),
            size=9, color=TXT, bold=True)
        txt(sl, desc,
            MC_L + Inches(0.12), mt + Inches(0.35),
            MC_W - Inches(0.24), Inches(0.24),
            size=8, color=TXT)
        txt(sl, note,
            MC_L + Inches(0.12), mt + Inches(0.54),
            MC_W - Inches(0.24), Inches(0.14),
            size=7, color=DG, italic=True)

    # ── Right column: evaluation pipeline + notice ───────────────────────
    RP = MC_L + MC_W + Inches(0.30)
    RW = CW - MC_W - Inches(0.30)

    txt(sl, "EVALUATION PIPELINE",
        RP, CT + Inches(0.14), RW, Inches(0.28),
        size=8.5, bold=True, color=N1, align=PP_ALIGN.CENTER)

    pipe_steps = [
        (TE,  "INPUT",    "D-Fire test-set images\n(potential benchmark dataset)"),
        (BL,  "INFER",    "YOLOv8-based model\ninference (640×640)"),
        (OR,  "FILTER",   "Confidence ≥ 0.25\nNMS (IoU = 0.45)"),
        (N1,  "COMPARE",  "Predicted bboxes vs.\nground-truth annotations"),
        (GR,  "METRICS",  "Compute Precision, Recall\nF1, mAP@0.5, mAP@0.5:0.95"),
    ]

    ps_h = Inches(0.70); ps_gap = Inches(0.12)
    ps_top = CT + Inches(0.50)

    for i, (col, label, body) in enumerate(pipe_steps):
        pt = ps_top + i * (ps_h + ps_gap)
        rbox(sl, RP, pt, RW, ps_h, fill=col)
        txt(sl, label,
            RP + Inches(0.12), pt + Inches(0.07),
            RW - Inches(0.24), Inches(0.24),
            size=9, bold=True, color=WH)
        txt(sl, body,
            RP + Inches(0.12), pt + Inches(0.33),
            RW - Inches(0.24), Inches(0.34),
            size=8, color=WH)
        if i < len(pipe_steps) - 1:
            cx = RP + RW / 2
            arrow_down(sl, cx, pt + ps_h, ps_gap - Inches(0.02),
                       color=DG, w=Pt(1.2))

    # Important notice
    note_top = ps_top + len(pipe_steps) * (ps_h + ps_gap) + Inches(0.08)
    note_h   = SH - FH - Inches(0.12) - note_top
    rbox(sl, RP, note_top, RW, note_h, fill=RGBColor(0xFF, 0xF5, 0xE0), border=OR, bw=1.2)
    txt(sl, "⚠  IMPORTANT ACADEMIC NOTE",
        RP + Inches(0.10), note_top + Inches(0.06),
        RW - Inches(0.20), Inches(0.24),
        size=8.5, bold=True, color=OR)
    txt(sl, "Quantitative benchmark results (mAP, Precision, Recall) will be "
            "reported after independent experimental evaluation against ground-truth "
            "annotated test data. No experimental metrics are presented at this stage.",
        RP + Inches(0.10), note_top + Inches(0.30),
        RW - Inches(0.20), note_h - Inches(0.36),
        size=7.5, color=TXT)


# ── SLIDE 8: CONTRIBUTION & ROADMAP ───────────────────────────────────────────
def s8_contribution(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, Inches(0), Inches(0), SW, SH, fill=LG)
    header(sl, "EXPECTED CONTRIBUTION & FUTURE ROADMAP", 8)
    footer(sl, 8)

    # ── Left: implemented ──────────────────────────────────────────────
    LW = CW * 0.52
    box(sl, LM, CT + Inches(0.14), LW, Inches(0.38), fill=GR)
    txt(sl, "✔  CURRENT CONTRIBUTIONS  (Implemented)",
        LM + Inches(0.12), CT + Inches(0.20), LW - Inches(0.24), Inches(0.26),
        size=9, bold=True, color=WH, align=PP_ALIGN.CENTER)

    contribs = [
        (GR, "Research",
             "•  Structured end-to-end fire and smoke detection methodology\n"
             "•  15-paper literature survey across 4 thematic areas\n"
             "•  Identified 5 research gaps addressed by this work"),
        (N1, "Technical",
             "•  YOLOv8-based inference pipeline (image / video / webcam)\n"
             "•  Dual-class detection: fire (class 1) and smoke (class 0)\n"
             "•  Confidence filtering (τ = 0.25) + NMS (IoU = 0.45)\n"
             "•  Severity heuristic: S = min(100, F×45 + Sm×20 + FPS×2)\n"
             "•  Four-tier urgency labels: critical / high / medium / low\n"
             "•  Rolling 30-frame FPS monitor"),
        (TE, "Engineering",
             "•  Modular Python codebase (src/detector.py, src/pipeline.py, run.py)\n"
             "•  CLI entry-point: --source, --conf, --device, --output\n"
             "•  Supports .mp4 / .avi / .mov / .mkv + live webcam"),
    ]

    cy = CT + Inches(0.60)
    for col, title, body in contribs:
        bh = body.count("\n") * Inches(0.24) + Inches(0.60)
        rbox(sl, LM, cy, LW, bh, fill=WH, border=col, bw=1.4)
        box(sl, LM, cy, Inches(0.06), bh, fill=col)
        txt(sl, title,
            LM + Inches(0.14), cy + Inches(0.07),
            LW - Inches(0.26), Inches(0.26),
            size=9.5, bold=True, color=col)
        txt(sl, body,
            LM + Inches(0.14), cy + Inches(0.33),
            LW - Inches(0.26), bh - Inches(0.40),
            size=8.5, color=TXT)
        cy += bh + Inches(0.12)

    # ── Right: roadmap ─────────────────────────────────────────────────
    RP  = LM + LW + Inches(0.28)
    RW  = CW - LW - Inches(0.28)
    box(sl, RP, CT + Inches(0.14), RW, Inches(0.38), fill=OR)
    txt(sl, "⏳  FUTURE ROADMAP  (Planned)",
        RP + Inches(0.12), CT + Inches(0.20), RW - Inches(0.24), Inches(0.26),
        size=9, bold=True, color=WH, align=PP_ALIGN.CENTER)

    phases = [
        (OR,  "Phase 1", "Experimental Evaluation",
               "Run inference on D-Fire test set\nMeasure Precision, Recall, F1, mAP@0.5"),
        (N1,  "Phase 2", "Comparative Study",
               "Benchmark vs. YOLOv5 and Faster R-CNN\nOn identical test data and hardware"),
        (BL,  "Phase 3", "Ablation Studies",
               "Vary τ_conf, imgsz, IoU threshold\nAnalyse impact on mAP and latency"),
        (PU,  "Phase 4", "Dataset Confirmation",
               "Verify D-Fire training provenance\nDocument data.yaml and class mappings"),
        (GR,  "Phase 5", "Reporting & Dissemination",
               "Compile results into final report\nPotential conference paper submission"),
    ]

    ph = Inches(0.80); pg = Inches(0.10)
    ry = CT + Inches(0.60)
    for col, phase, title, body in phases:
        rbox(sl, RP, ry, RW, ph, fill=WH, border=col, bw=1.4)
        box(sl, RP, ry, Inches(0.06), ph, fill=col)
        badge(sl, phase, RP + Inches(0.12), ry + Inches(0.24),
              Inches(0.82), Inches(0.26), fill=col, size=7.5, bold=True)
        txt(sl, title,
            RP + Inches(1.02), ry + Inches(0.07),
            RW - Inches(1.14), Inches(0.26),
            size=9.5, bold=True, color=col)
        txt(sl, body,
            RP + Inches(1.02), ry + Inches(0.34),
            RW - Inches(1.14), Inches(0.42),
            size=8, color=TXT)
        if ry + ph + pg < SH - FH - Inches(0.12):
            ry += ph + pg

    # Closing statement
    close_top = SH - FH - Inches(0.52)
    box(sl, LM, close_top, CW, Inches(0.40), fill=N1)
    txt(sl,
        "This project establishes a reproducible, modular baseline for automated fire and smoke detection "
        "using YOLOv8-based computer vision — a foundation for future multi-hazard, embedded, and edge-AI safety systems.",
        LM + Inches(0.16), close_top + Inches(0.06),
        CW - Inches(0.32), Inches(0.30),
        size=8, color=WH, align=PP_ALIGN.CENTER, italic=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.stdout.reconfigure(encoding="utf-8")
    OUT = "Fire_Smoke_Detection_Research_Presentation_FINAL.pptx"

    prs = new_prs()
    print("Building slides...")
    s1_title(prs)
    print("  [1/8] Title")
    s2_problem(prs)
    print("  [2/8] Problem Motivation")
    s3_literature(prs)
    print("  [3/8] Literature Review")
    s4_gap(prs)
    print("  [4/8] Research Gap")
    s5_objectives(prs)
    print("  [5/8] Research Objectives")
    s6_methodology(prs)
    print("  [6/8] Methodology")
    s7_evaluation(prs)
    print("  [7/8] Evaluation Framework")
    s8_contribution(prs)
    print("  [8/8] Contribution & Roadmap")

    prs.save(OUT)
    size_kb = os.path.getsize(OUT) // 1024
    print(f"\nSaved: {OUT}  ({size_kb} KB)")
    print(f"Slide count: {len(prs.slides)}")
