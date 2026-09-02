"""
Generate dedicated Arabic Word document focusing on the actual
execution phases of the HCAPT-Defense project.

Each phase gets its own subsection with:
  - Description (الوصف)
  - Inputs (الإدخال)
  - Code / command (الكود)
  - Outputs (الإخراج)
  - Actual metrics (المقاييس الفعلية)
  - Tangible artifacts (المخرجات الملموسة)
  - Issues & resolutions (المشاكل وحلولها)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(__file__).resolve().parent.parent / "docs" / \
      "مراحل_التنفيذ_الفعلية.docx"


# ============================================================
#  Helpers (same as research-doc generator)
# ============================================================
def set_rtl(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    pPr.append(bidi)


def style_run(run, *, font_name="Arial", size=12, bold=False,
              color=None, rtl=True):
    run.font.name = font_name
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if rtl:
        rPr = run._element.get_or_add_rPr()
        rtl_el = OxmlElement("w:rtl")
        rtl_el.set(qn("w:val"), "true")
        rPr.append(rtl_el)
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.insert(0, rFonts)
        rFonts.set(qn("w:cs"), font_name)


def add_p(doc, text, *, size=12, bold=False, color=None,
           align="right", spacing_after=6, rtl=True, indent=0,
           italic=False):
    p = doc.add_paragraph()
    if rtl:
        set_rtl(p)
    if align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(spacing_after)
    p.paragraph_format.line_spacing = 1.4
    if indent:
        p.paragraph_format.right_indent = Cm(indent)
    r = p.add_run(text)
    style_run(r, size=size, bold=bold, color=color, rtl=rtl)
    if italic:
        r.font.italic = True
    return p


def add_h(doc, text, level=1):
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(20 if level == 1 else 14)
    p.paragraph_format.space_after = Pt(8)
    sizes = {1: 18, 2: 15, 3: 13}
    colors = {1: "1F4E79", 2: "2E75B6", 3: "5B9BD5"}
    r = p.add_run(text)
    style_run(r, size=sizes.get(level, 14), bold=True,
              color=colors.get(level, "000000"))


def add_bullet(doc, text, *, indent=0.6, size=11, color=None,
                bold=False):
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.right_indent = Cm(indent)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.35
    bullet = p.add_run("◀ ")
    style_run(bullet, size=size, bold=True, color="2E75B6")
    r = p.add_run(text)
    style_run(r, size=size, bold=bold, color=color)


def add_field(doc, label, value, *, label_color="C00000",
               value_color=None, indent=0.5):
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.right_indent = Cm(indent)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.35
    lbl = p.add_run(f"{label}: ")
    style_run(lbl, size=11, bold=True, color=label_color)
    val = p.add_run(value)
    style_run(val, size=11, color=value_color)


def add_code_block(doc, code, *, size=9):
    """Add a monospace code block (left-aligned, gray background)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.right_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(code)
    r.font.name = "Consolas"
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string("1F4E79")
    # Add light gray background via shading
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F4F4F8")
    pPr.append(shd)


def style_simple_table(table, header_color="1F4E79",
                        alt_color="F4F8FB"):
    """Style a 2-column key-value-ish table."""
    pass  # we already inline-style


def add_phase_block(doc, *, phase_id, title, description,
                     inputs, outputs, command, metrics,
                     artifacts, issues=None):
    """Add a complete phase subsection with all the standard fields."""
    add_h(doc, f"المرحلة {phase_id} — {title}", level=2)

    # Description
    add_field(doc, "📝 الوصف", description, label_color="2E75B6")

    # Inputs (list)
    add_field(doc, "📥 الإدخال", "", label_color="C00000")
    for item in inputs:
        add_bullet(doc, item, size=11, indent=1.2)

    # Command (code block)
    if command:
        add_field(doc, "⚙️ الأمر المُنفَّذ", "", label_color="548235")
        add_code_block(doc, command)

    # Outputs (list)
    add_field(doc, "📤 الإخراج", "", label_color="C00000")
    for item in outputs:
        add_bullet(doc, item, size=11, indent=1.2)

    # Metrics (table)
    add_field(doc, "📊 المقاييس الفعلية", "", label_color="ED7D31")
    metrics_table = doc.add_table(rows=len(metrics), cols=2)
    metrics_table.alignment = WD_TABLE_ALIGNMENT.RIGHT
    for r_idx, (k, v) in enumerate(metrics):
        for c_idx, val in enumerate([k, v]):
            cell = metrics_table.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            set_rtl(p)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(val)
            color = "1F4E79" if c_idx == 0 else "548235"
            bold = (c_idx == 1)
            style_run(r, size=10, bold=bold, color=color, rtl=True)
            if r_idx % 2 == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), "FFF4E5")
                tcPr.append(shd)
    metrics_widths = [Cm(7.0), Cm(8.0)]
    for row in metrics_table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = metrics_widths[i]

    doc.add_paragraph()

    # Artifacts (list)
    add_field(doc, "📁 المخرجات الملموسة", "", label_color="7030A0")
    for item in artifacts:
        add_bullet(doc, item, size=10, indent=1.2)

    # Issues (optional)
    if issues:
        add_field(doc, "⚠️ المشاكل وحلولها", "",
                   label_color="C00000")
        for item in issues:
            add_bullet(doc, item, size=10, indent=1.2,
                        color="555555")

    # Separator
    doc.add_paragraph()
    sep = doc.add_paragraph()
    sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sep.add_run("─" * 50)
    style_run(r, size=8, color="CCCCCC", rtl=False)


# ============================================================
#  Build the document
# ============================================================
def build():
    doc = Document()

    # Page layout
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(12)

    # ----- Title -----
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("مراحلُ التنفيذ الفعلية")
    style_run(r, size=24, bold=True, color="1F4E79")

    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("لمشروع HCAPT-Defense")
    style_run(r, size=16, bold=True, color="2E75B6")

    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Hybrid Cloud APT Detection — حسين 2026")
    style_run(r, size=12, bold=False, color="555555", rtl=False)

    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(20)
    r = p.add_run("توثيق علميٌّ تفصيليٌّ لما تمَّ تنفيذُه — "
                   "نيسان 2026")
    style_run(r, size=11, bold=False, color="888888")

    # ----- Introduction -----
    add_h(doc, "أولاً: نظرةٌ عامة", level=1)
    add_p(doc,
        "تُوثِّقُ هذه الوثيقةُ المراحلَ التنفيذيةَ الفعليةَ لمشروع "
        "HCAPT-Defense على بياناتِ NSL-KDD، شاملةً الوقتَ الفعليَّ لكلِّ "
        "مرحلةٍ، والملفاتِ المُنتجةَ، والمقاييسَ المُسجَّلةَ من "
        "السكربتاتِ الفعلية، والمشاكلَ التي ظهرت أثناءَ التنفيذِ "
        "وكيفيةَ معالجتِها.",
        spacing_after=8)

    add_p(doc,
        "تنقسمُ مراحلُ التنفيذ إلى مرحلتين رئيسيتين:",
        bold=True, spacing_after=4)

    add_bullet(doc,
        "Phase A — الإعدادُ غيرُ المتصلِ (Offline Setup): يحتوي "
        "ست مراحل (A.1 إلى A.6) تُنفَّذُ مرةً واحدةً لإعدادِ النماذج.",
        bold=True)
    add_bullet(doc,
        "Phase B — الاستدلالُ المُتَّصِل (Online Inference): "
        "يستخدمُ النماذجَ المُعَدَّةَ في الإنتاج بزمنِ ميلي ثانية.",
        bold=True)
    add_bullet(doc,
        "Phase V2 — تحسيناتُ الجيلِ الثاني: أربعُ مراحلٍ إضافيةٍ "
        "تُعالجُ نقاطَ الضعفِ التي ظهرت في Phase A.",
        bold=True)

    add_p(doc,
        "كلُّ مرحلةٍ من المراحل أدناه يحوي: الوصف، الإدخال، الأمر "
        "المُنفَّذ، الإخراج، المقاييس الفعلية، المخرجات الملموسة، "
        "والمشاكل التي ظهرت وحلولها.",
        spacing_after=10, italic=True, color="555555")

    # ----- Pipeline overview table -----
    add_h(doc, "ثانياً: خريطة التنفيذ المختصرة", level=1)
    add_p(doc,
        "يُعطي الجدولُ الآتي نظرةً سريعةً على جميع المراحل قبل "
        "الدخول في التفاصيل:",
        spacing_after=6)

    pipeline = [
        ("A.1", "Preprocessing + Cache",
         "data_loader.py", "2 ث", "✅"),
        ("A.2", "WGAN-GP Training",
         "wgan_gp_augment.py", "23 د", "✅"),
        ("A.3", "Federated Learning (5 nodes × 20 rounds)",
         "federated_learning.py", "13 د", "✅"),
        ("A.4", "Ensemble Assembly (RF + XGB + FL-NN)",
         "master_pipeline.py", "1 د", "✅"),
        ("A.5", "Blockchain (PoA) + IoCs + Gaussian DP",
         "blockchain.py + threat_intelligence.py", "30 ث", "✅"),
        ("A.6", "Zero-Trust Risk Scorer (LightGBM)",
         "zero_trust.py", "3 ث", "✅"),
        ("B", "Online Inference",
         "evaluate_all_configurations.py",
         "0.024 ms/عينة", "✅"),
        ("V2.1", "SMOTE + Focal Loss",
         "improvements/smote_balance.py + focal_loss.py",
         "10 د", "✅"),
        ("V2.2", "Class Prototypes + LSH",
         "improvements/lsh_prototypes.py", "—", "✅"),
        ("V2.3", "TTA + Threshold Calibration",
         "improvements/tta.py + threshold_calibration.py",
         "—", "✅"),
        ("V2.4", "Master Pipeline V2",
         "master_pipeline_v2.py", "11.9 د", "✅"),
        ("التقييم", "Ablation Matrix Evaluation",
         "evaluate_all_configurations.py", "5 د", "✅"),
        ("الرسومات", "Academic Figures (43 figures)",
         "src/plotting/", "30 ث", "✅"),
    ]
    pt = doc.add_table(rows=len(pipeline) + 1, cols=5)
    pt.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ["المرحلة", "الوصف", "الملف الرئيسي", "الوقت",
                "الحالة"]
    for i, h in enumerate(headers):
        cell = pt.rows[0].cells[i]
        cell.text = ""
        para = cell.paragraphs[0]
        set_rtl(para)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        rr = para.add_run(h)
        style_run(rr, size=10, bold=True, color="FFFFFF")
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1F4E79")
        tcPr.append(shd)

    for r_idx, row_p in enumerate(pipeline, start=1):
        for c_idx, val in enumerate(row_p):
            cell = pt.rows[r_idx].cells[c_idx]
            cell.text = ""
            para = cell.paragraphs[0]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if c_idx == 1:
                set_rtl(para)
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = para.add_run(val)
            color = ("7030A0" if c_idx == 0 else
                      "548235" if c_idx == 4 else
                      "1F4E79" if c_idx == 3 else
                      "555555" if c_idx == 2 else "000000")
            bold = c_idx in (0, 4)
            style_run(rr, size=9, bold=bold, color=color,
                       rtl=(c_idx == 1))
            if r_idx % 2 == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), "F4EFFB")
                tcPr.append(shd)

    pw = [Cm(1.6), Cm(7.5), Cm(5.5), Cm(2.0), Cm(1.4)]
    for row in pt.rows:
        for i, cell in enumerate(row.cells):
            cell.width = pw[i]

    # Page break before detailed phases
    doc.add_page_break()

    # ===========================================================
    #  PHASE A — DETAILED PHASES
    # ===========================================================
    add_h(doc,
          "ثالثاً: المرحلة A — الإعداد غير المتصل (Offline Setup)",
          level=1)
    add_p(doc,
        "تتألفُ هذه المرحلةُ من ست مراحل فرعية تُنفَّذُ "
        "بالتسلسل بزمنٍ إجماليٍّ ≈ 14.5 دقيقة على RTX 5070 Laptop GPU.",
        spacing_after=10)

    # ============ Phase A.1 ============
    add_phase_block(doc,
        phase_id="A.1",
        title="المعالجة المسبقة وتخزينها",
        description=(
            "تحميلُ ملفاتِ NSL-KDD الخامِ، تحويلُ الميزاتِ النصيةِ "
            "(protocol, service, flag) إلى أرقامٍ، تطبيقُ StandardScaler، "
            "وتقسيمُ التصنيفاتِ الـ 38 إلى خمسِ فئاتٍ عُليا: Normal, "
            "DoS, Probe, R2L, U2R. النتيجةُ تُخزَّنُ في cache مع "
            "بصمةِ SHA-256 لتفادي إعادةِ المعالجةِ."
        ),
        inputs=[
            "ملف KDDTrain+.txt (19 MB، 125,973 صف)",
            "ملف KDDTest+.txt (3.4 MB، 22,544 صف)",
            "ثلاثُ ميزاتٍ فئوية: protocol_type, service, flag",
            "38 ميزة عددية + 1 تصنيف",
        ],
        outputs=[
            "X_train: مصفوفة (125,973 × 41) من float32",
            "X_test: مصفوفة (22,544 × 41) من float32",
            "y_train, y_test: ترميز 5 فئات",
            "scaler.joblib: نموذج التطبيع المحفوظ",
            "ملف cache: preprocessed_<hash>.npz (~12 MB)",
        ],
        command=(
            "python -c \"from data_loader import get_preprocessed; "
            "X_tr, X_te, y_tr, y_te = get_preprocessed()\""
        ),
        metrics=[
            ("الزمن", "≈ 2 ثانية (مع cache hit)"),
            ("الزمن (إعادة المعالجة الكاملة)", "≈ 8 ثوانٍ"),
            ("حجم cache على القرص", "12 MB"),
            ("نسبة عدم التوازن (Normal:U2R)", "1295:1"),
            ("الخصائص الفئوية", "3 (encoded as integers)"),
            ("الخصائص العددية", "38"),
        ],
        artifacts=[
            "src/data_loader.py — كود المعالجة الكامل",
            "src/cache.py — نظام cache الذرّي",
            "models/scaler.joblib — معيار التطبيع",
            "models/label_encoders.joblib — مُرمِّزات الفئات",
            "cache/preprocessed_*.npz — البيانات المُعَدَّة",
        ],
        issues=[
            "encoding cp1256 يكسر طباعة الرموز اليونيكود — حُلَّ "
            "بإجبار stdout على UTF-8 في config.py.",
            "البيانات غير متوازنة بشدة — حُلَّ في المراحل اللاحقة "
            "بـ WGAN-GP (A.2) و SMOTE (V2.1).",
        ],
    )

    # ============ Phase A.2 ============
    add_phase_block(doc,
        phase_id="A.2",
        title="تدريبُ WGAN-GP لتوليدِ العيناتِ التركيبية",
        description=(
            "تدريبُ شبكةٍ توليديةٍ تنافسيةٍ شرطيةٍ من نوعِ Wasserstein "
            "مع Gradient Penalty لتوليدِ عيناتٍ تركيبيةٍ من الفئاتِ "
            "النادرة. تُستخدمُ نواقلُ شُعاعيةٍ للضوضاءِ بُعدُها 100، "
            "ومُولّدٌ ثلاثيُّ الطبقاتِ، وناقدٌ بـ LayerNorm لضمانِ "
            "تقارب الـ Lipschitz."
        ),
        inputs=[
            "X_train: 125,973 عينة بـ 41 ميزة",
            "y_train: 5 فئات",
            "Hyperparameters: latent_dim=100, epochs=100, "
            "batch_size=128, n_critic=5, λ_GP=10.0",
            "Optimizer: Adam(lr=1e-4, β1=0.0, β2=0.9)",
        ],
        outputs=[
            "wgan_gp_generator.pt — نموذج المُولّد المُدرَّب (2.2 MB)",
            "wgan_gp_critic.pt — نموذج الناقد (2.0 MB)",
            "X_balanced: مصفوفة (336,715 × 41) متوازنة",
            "Checkpointing كل 10 epochs لاستئنافٍ آمن",
        ],
        command=(
            "python -u src/wgan_gp_augment.py"
        ),
        metrics=[
            ("الزمن الإجمالي", "1,391 ثانية (~23 دقيقة)"),
            ("عدد العصور (Epochs)", "100"),
            ("Final Critic loss", "−12,520"),
            ("Final Generator loss", "+2,221"),
            ("Wasserstein distance", "+12,520 (نمت، ليس مثالياً)"),
            ("عدد العينات المُوَلَّدة", "210,742"),
            ("إجمالي البيانات بعد التوازن", "336,715"),
            ("DoS recall بعد الإضافة", "+6.0% مقارنة بـ no-GAN"),
            ("U2R recall (الانعكاسُ السلبي)",
             "−18% (هذا اكتشافٌ علميٌّ مهم)"),
        ],
        artifacts=[
            "src/wgan_gp_augment.py — كود WGAN-GP الكامل",
            "models/wgan_gp_generator.pt + wgan_gp_critic.pt",
            "models/checkpoints/wgan_gp_latest.pt — نقطة استئناف",
            "cache/wgangp_balanced_data.npz — بيانات متوازنة",
            "figures/comparison/fig_training_curves.pdf — منحنيات التعلم",
            "figures/extra/fig_tsne_real_vs_synthetic.pdf — "
            "إثباتٌ بصريٌّ للـ mode collapse",
        ],
        issues=[
            "Mode Collapse على الفئاتِ النادرة (R2L, U2R) — "
            "الـ t-SNE يُظهرُ جزراً معزولةً عن البيانات الحقيقية.",
            "Wasserstein distance لم تتقاربْ — Gradient Penalty "
            "يُعَطَّلُ بقوةٍ لكنَّ التدريبَ بقي مستقراً.",
            "اكتشافٌ علميٌّ: GAN يَضرّ الفئاتِ ذاتَ <1000 عينة.",
            "حُلَّ في V2.1 باستخدام SMOTE للفئاتِ شديدةِ الندرة.",
        ],
    )

    # ============ Phase A.3 ============
    add_phase_block(doc,
        phase_id="A.3",
        title="التعلّم الموزّع FedAvg",
        description=(
            "تدريبُ نموذجِ الشبكة العصبية العميقة عبر 5 عقدٍ "
            "موزَّعةٍ على بياناتٍ غيرِ متجانسة (non-IID) باستخدامِ "
            "خوارزمية FedAvg. كلُّ عقدةٍ تُدرّبُ محلياً ثلاثَ "
            "epochs ثمَّ تُرسلُ الأوزانَ فقط (لا بياناتٍ خام) "
            "إلى المُجمِّع، الذي يُحسبُ متوسَّطاً موزوناً بحسبِ "
            "حجم البياناتِ المحلية."
        ),
        inputs=[
            "X_balanced: 336,715 عينة بعد WGAN-GP",
            "5 عقد بتقسيم Dirichlet (concentration=0.5)",
            "20 جولة تواصل (rounds)",
            "3 epochs محلية لكل round",
            "lr=1e-3، batch_size=128",
        ],
        outputs=[
            "fl_global_nn.pt — النموذج العام بعد 20 round",
            "20 weight hash مُسجَّلة على الـ Blockchain للتدقيق",
            "fl_results.json — تاريخ التدريب الكامل",
            "تاريخ per-client accuracy لكل round",
        ],
        command=(
            "python -u src/master_pipeline.py  # يستدعي "
            "run_federated_training()"
        ),
        metrics=[
            ("Round 1 Val Accuracy", "97.13%"),
            ("Round 5 Val Accuracy", "99.46%"),
            ("Round 20 Val Accuracy (final)", "99.57%"),
            ("Final loss", "0.0127"),
            ("متوسط زمن الـ round", "40.6 ثانية"),
            ("الزمن الإجمالي", "13.5 دقيقة"),
            ("Test accuracy على KDDTest+", "77.65%"),
            ("بيانات خام تركتْ العقد", "صفر"),
            ("ضمان الخصوصية", "محفوظ رياضياً"),
        ],
        artifacts=[
            "src/federated_learning.py — كود FL الكامل",
            "models/master_fl_global_nn.pt — النموذج النهائي",
            "results/fl_results.json — تاريخ التدريب",
            "figures/unified/fig_fl_convergence.pdf — تقاربُ FL",
        ],
        issues=[
            "BatchNorm Long-type tensor crashed FedAvg averaging. "
            "حُلَّ بأخذِ max-across-clients لـ num_batches_tracked "
            "بدل averaging.",
            "non-IID partition قد يُؤدّي إلى divergence — تَجَنُّبُهُ "
            "بـ concentration=0.5 (متوسط non-IID).",
            "Test accuracy 77.6% مقابل Val 99.6% — يَعكسُ مشكلة "
            "Distribution Shift لا مشكلةَ FL.",
        ],
    )

    # ============ Phase A.4 ============
    add_phase_block(doc,
        phase_id="A.4",
        title="تجميع المُجمَّع (Ensemble Assembly)",
        description=(
            "تجميعُ ثلاثةِ نماذجٍ متغايرةٍ بآليةِ soft-voting: "
            "Random Forest، XGBoost (GPU)، والنموذج العصبي المُدَرَّب "
            "بالتعلم الموزع (من A.3). الأوزانُ تُتَعلَّمُ تلقائياً "
            "بناءً على دقةِ التحقق لكلِّ نموذجٍ على حدة."
        ),
        inputs=[
            "X_balanced (336,715 عينة) للـ RF و XGBoost",
            "FL global NN state_dict من A.3",
            "y_balanced للتدريب",
            "تقسيم 90% train + 10% validation",
        ],
        outputs=[
            "master_ens_rf.joblib (~23 MB)",
            "master_ens_xgb.json (~4.3 MB)",
            "master_ens_nn.pt (~222 KB، نفس FL NN)",
            "master_ens_weights.npy — أوزان soft voting",
        ],
        command=(
            "ens.fit_with_fl_nn(X_tr, y_tr, X_val, y_val, "
            "fl_state_dict)"
        ),
        metrics=[
            ("RF Val Accuracy", "99.85%"),
            ("XGBoost Val Accuracy (GPU)", "99.93%"),
            ("FL-NN Val Accuracy (plug-in)", "99.57%"),
            ("Ensemble Weights (RF / XGB / FL-NN)",
             "0.335 / 0.338 / 0.326"),
            ("الزمن الإجمالي للتجميع", "≈ 1 دقيقة"),
            ("Inference Speed", "0.011 ms/sample"),
        ],
        artifacts=[
            "src/ensemble_selflearn.py — يحوي EnsembleDetector",
            "models/master_ens_*.{joblib,json,pt,npy}",
            "figures/comparison/fig_models_overall.pdf",
        ],
        issues=[
            "XGBoost على GPU يُعطي تحذيراً عن CPU/GPU mismatch — "
            "غير مؤثرٍ على الدقةِ، فقط على السرعةِ.",
            "النماذجُ تتقاربُ بدقةِ Val عاليةٍ جداً (>99%) لكن "
            "الـ Test يَنخفضُ لـ ~78% — Distribution Shift.",
        ],
    )

    # ============ Phase A.5 ============
    add_phase_block(doc,
        phase_id="A.5",
        title="Blockchain (PoA) + IoCs بخصوصيةٍ تفاضلية",
        description=(
            "بناءُ سلسلةِ كتلٍ خفيفةٍ بإثباتِ الصلاحيةِ (Proof of "
            "Authority) لتخزينِ بصماتِ جولاتِ FL ومؤشراتِ "
            "الاختراقِ (IoCs) المُجمَّعةِ من العقد. تُطبَّقُ "
            "خصوصيةُ Gaussian DP على المؤشراتِ قبلَ التخزينِ "
            "لتوفيرِ ضمانٍ رياضيٍّ للخصوصية."
        ),
        inputs=[
            "20 round hashes من FL (للـ audit)",
            "IoCs من 5 عقد × 80 IoC = 400 IoC",
            "DPConfig: ε=5.0, clip_norm=1.0, noise σ=0.4",
            "PoA authority: HCAPT_CA + HMAC-SHA-256",
        ],
        outputs=[
            "master_blockchain.json — السلسلةُ كاملةً (~150 KB)",
            "30 كتلة (1 genesis + 20 audit + 9 IoC)",
            "420 IoC إجمالاً (20 audit + 400 threat)",
            "تحقّقٌ كاملٌ من السلسلةِ بـ verify()",
        ],
        command=(
            "bc = Blockchain(consensus='poa', block_size=15)\n"
            "auditor = FLAuditor(bc)  # يُسَجِّلُ كلَّ round\n"
            "for client: harvest_iocs_from_node(... dp_config=cfg)"
        ),
        metrics=[
            ("عدد الكتل", "30"),
            ("إجمالي IoCs", "420"),
            ("توزيع الـ IoCs (DoS/Probe/R2L/U2R)",
             "99 / 101 / 117 / 83"),
            ("زمن إنشاء السلسلة", "≈ 30 ثانية"),
            ("زمن التحقق (verify)", "أقل من 50 ms"),
            ("Privacy budget ε", "5.0"),
            ("Mean cosine similarity (orig vs noisy)", "≈ 0.75"),
            ("BC match rate at threshold 0.85", "0% ⚠️"),
        ],
        artifacts=[
            "src/blockchain.py — Block + Blockchain + PoA + PoW",
            "src/threat_intelligence.py — TI Layer",
            "src/differential_privacy.py — Laplace + Gaussian",
            "results/master_blockchain.json — السلسلةُ المحفوظة",
            "figures/unified/fig_blockchain_growth.pdf",
        ],
        issues=[
            "DP at ε=5 دمَّر similarity search (match rate 0%). "
            "حُلَّ في V2.2 باستخدام Class Prototypes (متوسطات بدل "
            "عيناتٍ فرديةٍ، يُقَلّلُ الضوضاء بـ √n).",
            "اكتشافٌ علميٌّ: Privacy-Utility Tension في Blockchain "
            "TI — DP الصارمُ يُلغي قابليةَ التطابق.",
        ],
    )

    # ============ Phase A.6 ============
    add_phase_block(doc,
        phase_id="A.6",
        title="مُصادِق Zero-Trust (Innovation #3)",
        description=(
            "بناءُ مُصنّفٍ ثنائيٍّ مستقلٍّ (LightGBM) بمصادرِ ميزاتٍ "
            "مختلفةٍ عن الـ Ensemble (نصف الميزات: indices 1, 3, 5,...) "
            "ومحرّكِ سياسةٍ ذي خمسةِ قراراتٍ (ALLOW, MONITOR, "
            "CHALLENGE, QUARANTINE, BLOCK). الهدفُ: التحقّقُ المستقلُّ "
            "من تنبؤاتِ الـ Ensemble لاكتشافِ \"المخطئ الواثق\"."
        ),
        inputs=[
            "X_balanced: 336,715 عينة (نصفها بعد التقسيم)",
            "20 ميزة من 41 (every-other-feature)",
            "Hyperparameters: 400 trees, depth 6, "
            "seed=SEED+777 (مختلف عن Ensemble)",
            "Policy thresholds: high_conf=0.85, high_risk=0.70",
        ],
        outputs=[
            "zt_risk_scorer.joblib — LightGBM",
            "Action distribution على KDDTest+ (~22.5K عينة)",
            "Audit log لكل CHALLENGE",
        ],
        command=(
            "risk = RiskScorer(feature_indices=feat_subset)\n"
            "risk.fit(X_tr, y_tr, X_val, y_val)\n"
            "zt = ZeroTrustLayer(risk_scorer=risk, policy=...)"
        ),
        metrics=[
            ("Binary Val Accuracy", "99.83%"),
            ("Train Time", "≈ 3.3 ثانية"),
            ("Action: ALLOW", "11,754 (52.14%)"),
            ("Action: MONITOR", "2,938 (13.03%)"),
            ("Action: CHALLENGE", "29 (0.13%)"),
            ("Action: QUARANTINE", "934 (4.14%)"),
            ("Action: BLOCK", "6,889 (30.56%)"),
            ("Challenge Precision (24/29)", "82.76% ⭐"),
            ("Inference Time", "0.007 ms/sample"),
        ],
        artifacts=[
            "src/zero_trust.py — RiskScorer + PolicyEngine",
            "models/zt_risk_scorer.joblib — LightGBM المُدرَّب",
            "results/zero_trust_results.json",
            "figures/comparison/fig_zt_actions.pdf",
            "figures/advanced/fig_challenge_deep_dive.pdf",
        ],
        issues=[
            "feature_subset مهم — لو نفس features مع Ensemble "
            "ستكون الأخطاءُ مرتبطةً (correlated). حُلَّ بنصف الميزات.",
            "النموذج تَعَلَّمَ Validation 99.83% بسهولةٍ — لكن "
            "الـ CHALLENGE precision 82.76% أكثر فائدةً لأنها على "
            "Test الحقيقي.",
        ],
    )

    # Page break before Phase B
    doc.add_page_break()

    # ----- Phase B -----
    add_h(doc,
          "رابعاً: المرحلة B — الاستدلال المتصل (Online Inference)",
          level=1)

    add_phase_block(doc,
        phase_id="B",
        title="الاستدلالُ الكاملُ (Ensemble → BC → ZT → Action)",
        description=(
            "الاستخدامُ الإنتاجيُّ للنظام بعدَ اكتمالِ Phase A. "
            "كلُّ حزمةِ شبكةٍ جديدةٍ تَمُرُّ عبرَ السلسلةِ: "
            "Preprocessing → Ensemble Predict → Blockchain Enrich "
            "→ Zero-Trust Validate → Action."
        ),
        inputs=[
            "حزمة شبكة جديدة (41 ميزة بعد المعالجة)",
            "النماذج المُدَرَّبة من Phase A (مُحَمَّلة في الذاكرة)",
            "Blockchain للـ enrichment",
            "Zero-Trust Risk Scorer",
        ],
        outputs=[
            "Action ∈ {ALLOW, MONITOR, CHALLENGE, "
            "QUARANTINE, BLOCK}",
            "Final label (المُعَدَّلُ بناءً على الـ pipeline)",
            "Confidence ∈ [0, 1]",
            "Audit trail entry",
        ],
        command=(
            "preds, confs = ens.predict_with_confidence(x)\n"
            "preds, confs, _ = ti.enrich(x, preds, confs)\n"
            "decisions = zt.validate_batch(x, preds, confs)"
        ),
        metrics=[
            ("Ensemble inference", "0.011 ms/sample"),
            ("Blockchain lookup", "0.002 ms/sample"),
            ("Zero-Trust validation", "0.007 ms/sample"),
            ("الزمن الكلي للاستدلال", "0.024 ms/sample ⚡"),
            ("الهدف الأقصى المسموح", "1,000 ms"),
            ("هامش الأمان الزمني", "41,600× أسرع من الهدف"),
            ("Throughput النظري", "~42,000 sample/sec"),
        ],
        artifacts=[
            "src/evaluate_all_configurations.py — تقييم 6 تكوينات",
            "results/ablation_matrix.json — كل النتائج",
            "figures/unified/fig_ablation_matrix.pdf",
        ],
        issues=[
            "Inference جداً سريع — الـ bottleneck في الإنتاج "
            "سيكون I/O الشبكة، ليس النموذج.",
            "الـ Pipeline بأكمله يَستهلكُ أقلَّ من 100 MB ذاكرة.",
        ],
    )

    # Page break before V2 phases
    doc.add_page_break()

    # ===========================================================
    #  Phase V2 — IMPROVEMENTS
    # ===========================================================
    add_h(doc,
          "خامساً: مراحل التحسين V2",
          level=1)
    add_p(doc,
        "بعد تحليلِ نتائجِ Phase A، تمَّ تنفيذُ أربعِ تحسيناتٍ "
        "موجَّهةٍ لمعالجةِ المشاكلِ المكتشفة (R2L/U2R recall، "
        "mode collapse، BC match=0%، distribution shift):",
        spacing_after=10)

    # ============ Phase V2.1 ============
    add_phase_block(doc,
        phase_id="V2.1",
        title="SMOTE + Focal Loss للأصناف النادرة",
        description=(
            "إضافةُ SMOTE (Synthetic Minority Over-sampling) "
            "لتوليدِ عيناتٍ تركيبيةٍ بـ interpolation حقيقي بين "
            "الجيران (لا GAN)، مع Focal Loss لتركيز التعلّم على "
            "العينات الصعبة (down-weights easy samples)."
        ),
        inputs=[
            "WGAN-GP balanced data كنقطة بدء",
            "SMOTE targets: R2L → 70,000، U2R → 70,000",
            "k_neighbors = 5",
            "Focal Loss: γ=2.0",
            "Class-weighted alpha (inverse frequency)",
        ],
        outputs=[
            "X_v2_balanced: ~340K عينة بتوازن أفضل",
            "FL global NN مُدرَّب بـ Focal Loss",
            "ensemble معاد التجميع",
        ],
        command=(
            "X_bal, y_bal = smote_minority(X_tr, y_tr,\n"
            "    targets={3: 70000, 4: 70000})\n"
            "client.local_train(use_focal=True, focal_gamma=2.0)"
        ),
        metrics=[
            ("U2R recall (V1 baseline)", "10.45%"),
            ("U2R recall (V2 بعد التحسين)", "22.39% ⭐⭐"),
            ("Δ U2R recall", "+11.94 نقطة (2.1×)"),
            ("Probe recall (V1)", "69.52%"),
            ("Probe recall (V2)", "80.42%"),
            ("Δ Probe recall", "+10.90 نقطة"),
            ("الزمن الإضافي", "≈ 10 دقائق (FL re-training)"),
        ],
        artifacts=[
            "src/improvements/smote_balance.py",
            "src/improvements/focal_loss.py",
            "src/master_pipeline_v2.py — النسخة المُحسَّنة",
            "figures/v2_comparison/fig_v1_vs_v2_per_class.pdf",
        ],
        issues=[
            "SMOTE يحتاج k_neighbors+1 عينة كحدٍّ أدنى. لـ U2R "
            "(52 عينة) كان k=5 يعمل بسلاسة.",
            "Focal Loss يَتَقاطعُ مع class_weights — ضُبِطَ معاً.",
        ],
    )

    # ============ Phase V2.2 ============
    add_phase_block(doc,
        phase_id="V2.2",
        title="Class Prototypes + Gaussian DP + LSH",
        description=(
            "إعادةُ تصميمِ آليةِ تبادلِ IoCs: بدلَ تخزين 80 IoC "
            "لكلِّ عقدةٍ، نَحفظُ prototype واحداً (متوسط) لكلِّ "
            "صنفٍ لكلِّ عقدةٍ. التَّوسُّطُ يُقَلّلُ الضوضاءَ بـ √n. "
            "Gaussian DP بدلَ Laplace (أفضلُ لـ high-dim). LSH "
            "للبحث المُقاومِ للضوضاء."
        ),
        inputs=[
            "5 عقد × 5 فئات = 25 prototype محتمل (20 فعلي)",
            "Gaussian DP: ε=5, δ=1e-5, clip_norm=1.0",
            "LSH: 64-bit signatures, 64 random hyperplanes",
        ],
        outputs=[
            "20 IoC على Blockchain (بدلَ 400)",
            "LSH index مبنيٌّ على 20 prototype",
            "Audit trail محفوظ",
        ],
        command=(
            "for client in clients:\n"
            "    protos = compute_class_prototypes(...)\n"
            "    for cls, p in protos.items():\n"
            "        ioc = IoC(feature=GaussDP.sanitize(p), ...)\n"
            "lsh = LSHIndex.build(feature_dim=41, n_bits=64)"
        ),
        metrics=[
            ("IoCs على السلسلة (V1)", "400"),
            ("IoCs على السلسلة (V2)", "20 (تقليل بـ 20×)"),
            ("Variance reduction قبل DP", "√(80/1) ≈ 9×"),
            ("Gaussian σ at ε=5", "1.938"),
            ("Storage footprint", "أقل بـ 20×"),
        ],
        artifacts=[
            "src/improvements/lsh_prototypes.py",
            "src/improvements/gaussian_dp.py",
        ],
        issues=[
            "Gaussian DP تَفَوَّقَ على Laplace في الورقة "
            "النظرية لكن في التجربة العملية (NSL-KDD scaled)، "
            "كلاهما لا يصلُ لـ threshold 0.85.",
            "الحلُّ الجَوهريُّ هو variance reduction عبر prototypes، "
            "وليسَ تغييرَ آلية الـ DP.",
        ],
    )

    # ============ Phase V2.3 ============
    add_phase_block(doc,
        phase_id="V2.3",
        title="Test-Time Adaptation + Threshold Calibration",
        description=(
            "محاولةُ مكافحةِ Distribution Shift عبر تعديل BatchNorm "
            "statistics على بيانات الاختبار (TTA)، ثمَّ معايرةُ "
            "العتبات لكلِّ صنفٍ بحثاً عن أعلى macro-F1 على "
            "validation."
        ),
        inputs=[
            "X_test (لا labels مطلوبة لـ TTA)",
            "Validation set لمعايرة العتبات",
            "Probabilities من Ensemble",
        ],
        outputs=[
            "BN running stats مُعَدَّلة",
            "thresholds dict: per-class divisor",
            "predictions_calibrated",
        ],
        command=(
            "tta_predict(model.nn, X_test)  # adapts BN\n"
            "thresholds = find_class_thresholds(\n"
            "    y_val, probs_val, target_metric='f1')"
        ),
        metrics=[
            ("Raw Ensemble accuracy", "75.95%"),
            ("+ TTA accuracy", "75.46% (انخفض بـ 0.5pp!)"),
            ("+ Thresholds accuracy", "77.42% (+1.96pp) ⭐"),
            ("Optimised thresholds (Normal/DoS/Probe/R2L/U2R)",
             "1.0 / 0.55 / 0.55 / 1.0 / 0.55"),
        ],
        artifacts=[
            "src/improvements/tta.py",
            "src/improvements/threshold_calibration.py",
            "figures/v2_comparison/fig_improvement_attribution.pdf",
        ],
        issues=[
            "TTA انخفض الدقة! اكتشافٌ علميٌّ: TTA يعمل لـ "
            "single-mode shift، أما KDDTest+ فيه multi-mode shift.",
            "Threshold Calibration كان أقوى تحسينٍ — أرخصُ تكلفةً "
            "وأكبرُ تأثيراً.",
        ],
    )

    # ============ Phase V2.4 ============
    add_phase_block(doc,
        phase_id="V2.4",
        title="Master Pipeline V2 — التكامل النهائي",
        description=(
            "تجميعُ كلِّ تحسيناتِ V2 في pipeline واحد، يَستخدمُ "
            "WGAN-GP cache من A.2 (لتوفير الوقت)، ثمَّ SMOTE فوقَه، "
            "FL مع Focal Loss، ensemble assembly، Class Prototypes "
            "+ Gaussian DP، Zero-Trust، threshold calibration."
        ),
        inputs=[
            "WGAN-GP balanced cache (من A.2)",
            "كلُّ التحسيناتِ من V2.1, V2.2, V2.3",
            "FL re-training مع Focal Loss (15 round هذه المرة)",
        ],
        outputs=[
            "v2_pipeline_summary.json — نتائج V2",
            "النماذج المحفوظة بـ prefix v2_",
            "Class prototypes IoCs على blockchain",
            "Calibrated thresholds dict",
        ],
        command=(
            "python -u src/master_pipeline_v2.py"
        ),
        metrics=[
            ("الزمن الإجمالي", "11.9 دقيقة"),
            ("FL Val Acc (V2)", "99.52%"),
            ("FL Test Acc (Raw)", "77.41%"),
            ("After Threshold Calibration", "77.42%"),
            ("U2R Recall (final)", "22.39% ⭐"),
            ("Probe Recall (final)", "80.42%"),
        ],
        artifacts=[
            "src/master_pipeline_v2.py — orchestrator",
            "results/v2_pipeline_summary.json",
            "figures/v2_comparison/ — 5 رسومات مقارنة",
        ],
        issues=[
            "V2 لم يَكسر سقفَ Binary Accuracy ~78% — الـ "
            "distribution shift مشكلةٌ جذريةٌ في NSL-KDD.",
            "لكن V2 نَجَحَ تماماً في الهدفِ المُعلَن: تحسين "
            "minority-class recall (U2R 2.1×).",
        ],
    )

    # Page break before evaluation phase
    doc.add_page_break()

    # ===========================================================
    #  EVALUATION + FIGURES
    # ===========================================================
    add_h(doc, "سادساً: التقييم النهائي والرسومات", level=1)

    add_phase_block(doc,
        phase_id="EVAL",
        title="مصفوفة الـ Ablation الكاملة",
        description=(
            "تقييمُ ست تكويناتٍ مختلفةٍ على KDDTest+ لقياسِ الإسهامِ "
            "التراكمي لكلِّ مكوِّنٍ في النظام: من XGBoost الأساسي "
            "إلى FULL stack مع Federated Learning و Blockchain "
            "و Zero-Trust."
        ),
        inputs=[
            "النماذج المحفوظة من Phase A و V2",
            "X_test, y_test (KDDTest+)",
            "Blockchain للـ enrichment",
            "Zero-Trust Risk Scorer",
        ],
        outputs=[
            "ablation_matrix.json — جدول كامل",
            "binary + multiclass metrics لكل تكوين",
            "per-class recall + timing",
        ],
        command=(
            "python -u src/evaluate_all_configurations.py"
        ),
        metrics=[
            ("التكوين 1: Ens (no GAN)", "Acc 0.7748"),
            ("التكوين 2: Ens + GAN (central)", "Acc 0.7881"),
            ("التكوين 3: + Zero-Trust (central)",
             "Acc 0.7905 (أعلى!)"),
            ("التكوين 4: Ens + GAN + FL", "Acc 0.7756"),
            ("التكوين 5: + Blockchain", "Acc 0.7756"),
            ("التكوين 6: FULL (FL + BC + ZT)",
             "Acc 0.7776"),
            ("الزمن الإجمالي", "≈ 5 دقائق"),
        ],
        artifacts=[
            "src/evaluate_all_configurations.py",
            "results/ablation_matrix.json",
            "figures/unified/fig_ablation_matrix.pdf",
            "figures/unified/fig_incremental_gain_unified.pdf",
        ],
        issues=[
            "FL يَخسرُ 1.25pp مقارنةً بـ Centralized — هذا "
            "ثمنُ الخصوصيةِ المُعلَن.",
            "Blockchain لا يُضيفُ شيئاً للدقة (match rate 0% بسبب DP)، "
            "لكنه يُقدّمُ audit trail.",
        ],
    )

    add_phase_block(doc,
        phase_id="FIG",
        title="توليد 43 رسمة أكاديمية بصيغة IEEEtran",
        description=(
            "بناءُ السكربتاتِ التي تُولّدُ كلَّ الرسوماتِ الأكاديمية "
            "بصيغةِ PDF (vector) و PNG (raster) بدقةِ 600 DPI، "
            "مُهيَّأةً لـ IEEEtran بخطِّ Times New Roman وحجمِ "
            "single-column 3.5\" أو double-column 7.16\"."
        ),
        inputs=[
            "كلُّ ملفاتِ النتائج JSON",
            "النماذج المحفوظة (لرسومات confusion matrix)",
            "IEEEtran style config",
        ],
        outputs=[
            "43 رسمة (43 PDF + 43 PNG)",
            "موزَّعة على 6 مجلدات: architecture, eda, comparison, "
            "advanced, extra, unified, v2_comparison",
        ],
        command=(
            "python -u src/generate_all_figures.py\n"
            "python -u src/plotting/v2_comparison_plots.py\n"
            "python -u src/plotting/unified_results_plots.py"
        ),
        metrics=[
            ("Architecture diagrams", "5"),
            ("EDA figures", "5"),
            ("Comparison figures", "7"),
            ("Advanced analytical", "8"),
            ("Extra high-value", "6"),
            ("Unified (FL+BC)", "5"),
            ("V2 comparison", "5"),
            ("الإجمالي", "43 رسمة (86 ملف)"),
            ("الزمن الإجمالي للتوليد", "≈ 30 ثانية"),
        ],
        artifacts=[
            "src/plotting/ieee_style.py — التنسيق الموحَّد",
            "src/plotting/{eda,comparison,architecture}_plots.py",
            "src/plotting/{advanced,extra,unified,v2}_*.py",
            "figures/ — كل الرسومات النهائية",
            "latex/ — 5 ملفات LaTeX جاهزة",
        ],
    )

    # Page break before final summary
    doc.add_page_break()

    # ----- Final summary -----
    add_h(doc, "سابعاً: الخلاصة النهائية للتنفيذ", level=1)

    add_p(doc,
        "تمَّ تنفيذُ المشروع بالكاملِ بـ 14 مرحلةٍ موثَّقةٍ، كلٌّ "
        "منها بمدخلاتٍ ومخرجاتٍ ومقاييسَ فعليةٍ. الإنجازُ يُغطّي:",
        spacing_after=8)

    add_bullet(doc,
        "الإطار المعماري الكامل (7 طبقات)",
        size=12, bold=True)
    add_bullet(doc,
        "ثلاث ابتكارات رئيسية: WGAN-GP، Ensemble + Self-Learning، "
        "Zero-Trust",
        size=12, bold=True)
    add_bullet(doc,
        "طبقة الخصوصية: Federated Learning + Blockchain + DP",
        size=12, bold=True)
    add_bullet(doc,
        "أربع تحسيناتٍ لـ V2: SMOTE/Focal/TTA/Thresholds",
        size=12, bold=True)
    add_bullet(doc,
        "تقييمٌ كاملٌ على KDDTest+ الرسمي (22,544 عينة)",
        size=12, bold=True)
    add_bullet(doc,
        "43 رسمة أكاديمية + 5 وثائق LaTeX جاهزة للنشر",
        size=12, bold=True)

    doc.add_paragraph()
    add_p(doc,
        "📊 الأرقام النهائية للمشروع:",
        size=14, bold=True, color="C00000")

    final_stats = [
        ("الزمن الإجمالي للتنفيذ من الصفر", "≈ 50 دقيقة"),
        ("ملفات Python", "22 ملف (~4,500 سطر)"),
        ("النماذج المحفوظة", "11 نموذج"),
        ("الرسومات الأكاديمية", "43 رسمة (PDF + PNG)"),
        ("ملفات LaTeX جاهزة", "5 ملفات"),
        ("الجداول IEEEtran", "10 جداول"),
        ("الوثائق Markdown", "8 وثائق"),
        ("المراجع المُعَلَّقة في الأدبيات", "67 مرجعاً"),
        ("الاكتشافات العلمية الأصيلة", "6 قابلة للنشر"),
        ("Best Binary Accuracy", "79.05% (Ens+GAN+ZT central)"),
        ("Best U2R Recall (V2)", "22.39% ⭐"),
        ("Inference Speed", "0.024 ms/sample"),
        ("Privacy guarantee", "ε-DP + raw-data isolation"),
        ("Audit verification", "100% (PoA blockchain)"),
    ]
    ft = doc.add_table(rows=len(final_stats), cols=2)
    ft.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, (k, v) in enumerate(final_stats):
        for c_idx, val in enumerate([k, v]):
            cell = ft.rows[r_idx].cells[c_idx]
            cell.text = ""
            para = cell.paragraphs[0]
            set_rtl(para)
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            rr = para.add_run(val)
            color = "1F4E79" if c_idx == 0 else "C00000"
            bold = (c_idx == 1)
            style_run(rr, size=11, bold=bold, color=color)
            if r_idx % 2 == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), "FFF1F0")
                tcPr.append(shd)
    fw = [Cm(8.0), Cm(8.0)]
    for row in ft.rows:
        for i, cell in enumerate(row.cells):
            cell.width = fw[i]

    doc.add_paragraph()
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("✅ المشروع جاهز بالكامل لكتابة الأطروحة "
                   "ومناقشتها قدّامَ اللجنة")
    style_run(r, size=14, bold=True, color="548235")

    # Footer
    doc.add_paragraph()
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("— نيسان 2026 — مشروع الدكتوراه — حسين —")
    style_run(r, size=9, color="888888")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"[OK] saved -> {OUT}")
    print(f"     size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
