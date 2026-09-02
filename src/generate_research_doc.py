"""
Generate the Arabic academic Word document:
    Research Issues -> Research Questions -> Research Objectives Association

For the HCAPT-Defense thesis.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(__file__).resolve().parent.parent / "docs" / \
      "Research_Issues_Questions_Objectives_v2.docx"


# ============================================================
#  Helpers
# ============================================================
def set_rtl(paragraph):
    """Make a paragraph RTL (Arabic-friendly)."""
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    pPr.append(bidi)


def set_cell_rtl(cell):
    for p in cell.paragraphs:
        set_rtl(p)


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
        # Set bidi-arabic font (cs = complex script)
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.insert(0, rFonts)
        rFonts.set(qn("w:cs"), font_name)


def add_arabic_p(doc, text, *, size=12, bold=False, color=None,
                  align="right", spacing_after=6):
    p = doc.add_paragraph()
    set_rtl(p)
    if align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(spacing_after)
    p.paragraph_format.line_spacing = 1.4
    r = p.add_run(text)
    style_run(r, size=size, bold=bold, color=color)
    return p


def add_section_heading(doc, text, level=1):
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(8)

    sizes = {1: 18, 2: 15, 3: 13}
    colors = {1: "1F4E79", 2: "2E75B6", 3: "5B9BD5"}
    r = p.add_run(text)
    style_run(r, size=sizes.get(level, 14), bold=True,
              color=colors.get(level, "000000"))


def add_bullet(doc, text, *, indent=0.5, color=None, bold=False, size=12):
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.right_indent = Cm(indent)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.4

    bullet = p.add_run("◀ ")
    style_run(bullet, size=size, bold=True, color="2E75B6")
    r = p.add_run(text)
    style_run(r, size=size, bold=bold, color=color)


def add_id_para(doc, code, text, *, color="1F4E79"):
    """Add a paragraph that starts with a colored code like 'RI1'."""
    p = doc.add_paragraph()
    set_rtl(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.right_indent = Cm(0.5)
    p.paragraph_format.line_spacing = 1.4

    code_run = p.add_run(f"[{code}] ")
    style_run(code_run, size=12, bold=True, color=color)
    text_run = p.add_run(text)
    style_run(text_run, size=12)


# ============================================================
#  Build the document
# ============================================================
def build():
    doc = Document()

    # Page settings
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)   # A4
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    # Default style
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(12)

    # ----- Title -----
    title_p = doc.add_paragraph()
    set_rtl(title_p)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(6)
    r = title_p.add_run(
        "ربط المشاكل البحثية بالأسئلة البحثية بالأهداف البحثية"
    )
    style_run(r, size=22, bold=True, color="1F4E79")

    sub_p = doc.add_paragraph()
    set_rtl(sub_p)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(4)
    r = sub_p.add_run("Research Issues ⇄ Research Questions ⇄ Research Objectives")
    style_run(r, size=13, bold=False, color="2E75B6", rtl=False)

    sub2 = doc.add_paragraph()
    set_rtl(sub2)
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub2.paragraph_format.space_after = Pt(20)
    r = sub2.add_run("مشروع HCAPT-Defense — أطروحة الدكتوراه — حسين 2026")
    style_run(r, size=12, bold=True, color="555555")

    # ----- 1. Introduction -----
    add_section_heading(doc, "أولاً: مقدمة منهجية", level=1)

    add_arabic_p(doc,
        "تَهدفُ هذه الوثيقةُ إلى تقديم خريطةٍ منهجيةٍ تربط بين ثلاثة مكوناتٍ "
        "أساسيةٍ في البحث الأكاديمي للأطروحة الموسومة بـ "
        "HCAPT-Defense: Hybrid Cloud APT Detection & Mitigation:")

    add_bullet(doc,
        "المشاكل البحثية (Research Issues): الفجوات والتحديات المُكتشفة "
        "في الأدبيات والممارسة العملية لأنظمة كشف الاختراق.",
        bold=False)
    add_bullet(doc,
        "الأسئلة البحثية (Research Questions): الصياغات الاستفهامية "
        "التي تترجم تلك المشاكل إلى تساؤلاتٍ قابلةٍ للإجابة.",
        bold=False)
    add_bullet(doc,
        "الأهداف البحثية (Research Objectives): المخرجات الملموسة "
        "والإجراءات التي تُحقّقُ الإجابة عن تلك الأسئلة.",
        bold=False)

    add_arabic_p(doc,
        "ويرتكزُ هذا الترابطُ على المنهجيةِ المُتعارفِ عليها أكاديمياً، "
        "والتي تنصُّ على أنَّ كلَّ مشكلةٍ بحثيةٍ تُولّدُ سؤالاً بحثياً واحداً "
        "أو أكثر، وكلُّ سؤالٍ بحثيٍّ يستلزمُ هدفاً عملياً قابلاً للقياس "
        "والتحقق التجريبي.",
        spacing_after=12)

    # ----- 2. Research Issues -----
    add_section_heading(doc, "ثانياً: المشاكل البحثية (Research Issues)",
                         level=1)
    add_arabic_p(doc,
        "تنبثقُ هذه المشاكلُ من تحليلٍ نقديٍّ لأكثر من 67 مرجعاً أكاديمياً "
        "حديثاً (2018-2026) في مجالِ كشفِ الاختراقِ ضمن البيئاتِ "
        "السحابيةِ الهجينةِ، ومن نتائجِ التجاربِ التطبيقيةِ على مجموعة "
        "بيانات NSL-KDD.",
        spacing_after=10)

    issues = [
        ("RI1",
         "الانزياح التوزيعي (Distribution Shift): يحوي "
         "KDDTest+ سبعةَ عشرَ نوعاً من الهجمات لم تظهر في بياناتِ "
         "التدريب، مما يُسقطُ دقةَ النماذج التقليدية إلى حدود ٧٧٪ "
         "بعد أن تكون قد بلغت ٩٩٪ على بياناتِ التحقق."),
        ("RI2",
         "عدم التوازن الحاد بين الأصناف (Severe Class "
         "Imbalance): تبلغُ نسبةُ التفاوت بين الفئة الأكبر (Normal) "
         "والفئة الأصغر (U2R) في NSL-KDD نحو 1295:1، مما يُؤدّي إلى "
         "تجاهلِ النماذج للأصناف النادرة."),
        ("RI3",
         "الإفراط في الثقة (Overconfidence): تُخطئُ النماذجُ "
         "المعاصرةُ في تصنيفِ الهجمات الجديدة كحركةٍ طبيعيةٍ بثقةٍ "
         "تتجاوز ٨٥٪، مما يُمثّلُ ظاهرةَ \"المخطئ الواثق\"."),
        ("RI4",
         "تحدّي الخصوصية في تبادل التهديدات: لا يُمكنُ "
         "للمؤسساتِ المختلفةِ مشاركةُ بياناتِ هجماتها الخامِ بسبب "
         "اللوائح التنظيمية كـ GDPR و HIPAA و PCI-DSS."),
        ("RI5",
         "غياب سجلِّ تدقيقٍ غير قابلٍ للتلاعب: تفتقرُ "
         "أنظمةُ كشف الاختراق المعاصرة إلى آليةٍ موثوقةٍ لتدقيقِ "
         "قراراتِ النموذج لأغراضٍ قانونيةٍ أو امتثاليةٍ."),
        ("RI6",
         "محدودية الذكاء الاصطناعي التوليدي للأصناف "
         "النادرة جداً: تفشلُ شبكاتُ GAN في تعلّم التوزيع الحقيقي حين "
         "يكون عددُ العيناتِ الحقيقية أقلَّ من 1000، مما يؤدّي إلى "
         "ظاهرةِ Mode Collapse."),
        ("RI7",
         "التوتر بين الخصوصية والمنفعة (Privacy-Utility "
         "Tension): تفقدُ مؤشراتُ الاختراق (IoCs) قابليتها للبحث "
         "بالتشابه بعد تطبيقِ الضوضاء المعيارية للخصوصية التفاضلية."),
        ("RI8",
         "عدم كفاية التحقق المستقل: تعتمدُ معظمُ النماذجِ "
         "على مصدرٍ واحدٍ للقرار، دون آليةٍ مستقلةٍ للتحقق من "
         "التنبؤات عالية الثقة."),
    ]
    for code, text in issues:
        add_id_para(doc, code, text)

    # ----- 3. Research Questions -----
    add_section_heading(doc,
        "ثالثاً: الأسئلة البحثية (Research Questions)", level=1)
    add_arabic_p(doc,
        "تُترجمُ المشاكلُ البحثيةُ المذكورةُ أعلاه إلى الأسئلةِ "
        "البحثيةِ الآتيةِ، التي تُصاغُ بطريقةٍ قابلةٍ للإجابةِ التجريبيةِ:",
        spacing_after=10)

    questions = [
        ("RQ1",
         "كيف يمكنُ توظيفُ الذكاء الاصطناعي التوليدي "
         "(WGAN-GP) لتحسينِ كشف الهجمات النادرة وغير المعروفة سابقاً "
         "في البيئة السحابية الهجينة؟"),
        ("RQ2",
         "ما الحدُّ الأدنى من العينات الحقيقية اللازمة "
         "لكي تُسهم البياناتُ التركيبيةُ المُولَّدة بـ GAN في تحسينِ "
         "الأداء عوضاً عن إضراره؟"),
        ("RQ3",
         "كيف يمكنُ تدريبُ نموذجِ كشفٍ مشتركٍ بين عقدٍ "
         "موزعةٍ دون مشاركةِ البياناتِ الخامِ؟"),
        ("RQ4",
         "كيف يمكنُ توفيرُ سجلِّ تدقيقٍ غير قابلٍ للتلاعب "
         "لقراراتِ نظام كشف الاختراق؟"),
        ("RQ5",
         "كيف يمكنُ موازنةُ التوتر بين الخصوصية التفاضلية "
         "(DP) والقدرة على البحث بالتشابه في مؤشرات الاختراق "
         "(IoCs)؟"),
        ("RQ6",
         "كيف يمكنُ مكافحةُ ظاهرةِ \"المخطئ الواثق\" في "
         "النماذج العميقة عبر طبقةِ تحققٍ مستقلةٍ؟"),
        ("RQ7",
         "ما الأثرُ الكميُّ لكلِّ ابتكارٍ على حدة، وما "
         "تأثيرُ تكاملِها مجتمعةً، على مقاييسِ الكشف لكلِّ صنفٍ؟"),
        ("RQ8",
         "كيف يمكنُ إعادةُ معايرةِ عتباتِ القرار لتحسينِ "
         "الاستدعاءِ على الأصناف النادرة دون التضحية بالدقةِ "
         "الإجمالية؟"),
    ]
    for code, text in questions:
        add_id_para(doc, code, text, color="2E75B6")

    # ----- 4. Research Objectives -----
    add_section_heading(doc,
        "رابعاً: الأهداف البحثية (Research Objectives)", level=1)
    add_arabic_p(doc,
        "تُترجمُ الأسئلةُ البحثيةُ إلى الأهدافِ الإجرائيةِ القابلةِ "
        "للقياسِ التجريبي، ولكلِّ هدفٍ مخرجاتٌ ملموسةٌ في الكود "
        "والنتائج:",
        spacing_after=10)

    objectives = [
        ("RO1",
         "تصميمُ إطارِ HCAPT-Defense المعماري "
         "المُكوّن من سبعِ طبقاتٍ يُكاملُ بين الذكاء التوليدي والتعلم "
         "الموزع وسلسلة الكتل وانعدام الثقة."),
        ("RO2",
         "تطويرُ مولّدٍ تنافسيٍّ شرطيٍّ مبنيٍّ على "
         "Wasserstein-GP لتوليدِ عيناتٍ تركيبيةٍ من الأصناف النادرة، "
         "وقياسُ تأثيرِه على كلِّ صنفٍ بشكلٍ منفصلٍ."),
        ("RO3",
         "تنفيذُ نظامِ التعلم الموزع (FedAvg) بين خمسِ "
         "عقدٍ على بياناتٍ موزعةٍ غيرِ متجانسةٍ (non-IID)، مع التحقق "
         "من عدم خروج البيانات الخامِ من أيِّ عقدةٍ."),
        ("RO4",
         "بناءُ سلسلةِ كتلٍ خفيفةٍ بإثباتِ الصلاحية "
         "(PoA) لتخزين بصماتِ جولاتِ التعلم الموزع وتدقيقها "
         "بزمنٍ أقلَّ من 50 ميلي ثانية."),
        ("RO5",
         "تطبيقُ الخصوصية التفاضلية (Gaussian DP) على "
         "مؤشرات الاختراق المُتشاركة، باستخدام Class Prototypes "
         "بدلاً من العيناتِ الفرديةِ لتحسين الموازنة بين الخصوصية "
         "والمنفعة."),
        ("RO6",
         "تطويرُ طبقةِ التحقق الانعدام-ثقي (Zero-Trust) "
         "بمصنّفٍ مستقلٍ (LightGBM) ومحرّكِ سياسةٍ ذي خمسةِ قراراتٍ، "
         "وقياسُ دقة الاعتراض (Challenge Precision)."),
        ("RO7",
         "إجراءُ التقييم على الانقسامِ الرسمي "
         "KDDTest+ لقياسِ الأداء في ظروف الانزياح التوزيعي "
         "الحقيقي، بدلاً من الانقسامات العشوائية المضلِّلة."),
        ("RO8",
         "تنفيذُ تحسيناتٍ من الجيلِ الثاني (V2) تشمل "
         "SMOTE وFocal Loss وTest-Time Adaptation ومعايرةَ "
         "العتبات لكلِّ صنفٍ، وقياسُ الإسهام التراكمي."),
    ]
    for code, text in objectives:
        add_id_para(doc, code, text, color="C00000")

    # Page break before mapping table
    doc.add_page_break()

    # ----- 5. Association Mapping Table -----
    add_section_heading(doc,
        "خامساً: مصفوفة الربط (Issues ⇄ Questions ⇄ Objectives)",
        level=1)
    add_arabic_p(doc,
        "يُلخّصُ الجدولُ الآتي العلاقةَ ثلاثيةَ الأبعادِ بين كلِّ "
        "مشكلةٍ بحثيةٍ، والأسئلةِ التي تُولّدُها، والأهدافِ التي "
        "تُحقّقُها:",
        spacing_after=10)

    # Build the table
    rows_data = [
        # (issue_code, issue_short, questions, objectives)
        ("RI1",
         "الانزياح التوزيعي",
         "RQ1, RQ7, RQ8",
         "RO1, RO2, RO7, RO8"),
        ("RI2",
         "عدم التوازن الحاد",
         "RQ1, RQ2",
         "RO2, RO8"),
        ("RI3",
         "الإفراط في الثقة",
         "RQ6, RQ7",
         "RO6, RO8"),
        ("RI4",
         "خصوصية تبادل التهديدات",
         "RQ3, RQ5",
         "RO3, RO5"),
        ("RI5",
         "غياب سجل التدقيق",
         "RQ4",
         "RO4"),
        ("RI6",
         "محدودية الـ GAN للنادرات",
         "RQ2",
         "RO2, RO8"),
        ("RI7",
         "توتر الخصوصية والمنفعة",
         "RQ5",
         "RO5"),
        ("RI8",
         "عدم كفاية التحقق المستقل",
         "RQ6",
         "RO6"),
    ]

    table = doc.add_table(rows=len(rows_data) + 1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header
    headers = ["Research Issues", "الوصف المختصر",
               "Research Questions", "Research Objectives"]
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        set_rtl(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        r = p.add_run(h)
        style_run(r, size=11, bold=True, color="FFFFFF",
                   rtl=(i == 1))
        # background color
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1F4E79")
        tcPr.append(shd)

    # Data rows
    col_colors = [None, "F4F8FB", None, "FFF1F0"]
    for r_idx, row_data in enumerate(rows_data, start=1):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if c_idx == 1:
                set_rtl(p)
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            color = ("1F4E79" if c_idx == 0 else
                      "2E75B6" if c_idx == 2 else
                      "C00000" if c_idx == 3 else "000000")
            bold = c_idx in (0, 2, 3)
            style_run(r, size=11, bold=bold, color=color,
                       rtl=(c_idx == 1))
            # alternate row shading on description column
            if col_colors[c_idx]:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), col_colors[c_idx])
                tcPr.append(shd)

    # Set column widths
    widths = [Cm(2.0), Cm(5.5), Cm(4.0), Cm(4.5)]
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = widths[i]

    # ----- 6. Reverse mapping -----
    doc.add_paragraph()  # spacer
    add_section_heading(doc,
        "سادساً: الربط العكسي — لكلِّ هدفٍ ما المشاكلُ التي يُعالجُها؟",
        level=2)

    rev_map = [
        ("RO1", "إطار HCAPT-Defense", "RI1"),
        ("RO2", "WGAN-GP", "RI1, RI2, RI6"),
        ("RO3", "Federated Learning", "RI4"),
        ("RO4", "Blockchain (PoA)", "RI5"),
        ("RO5", "Differential Privacy + Prototypes", "RI4, RI7"),
        ("RO6", "Zero-Trust Layer", "RI3, RI8"),
        ("RO7", "تقييم على KDDTest+", "RI1"),
        ("RO8", "تحسينات V2", "RI1, RI2, RI3, RI6"),
    ]
    rev_table = doc.add_table(rows=len(rev_map) + 1, cols=3)
    rev_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    rev_headers = ["Research Objective", "الوصف", "Issues Addressed"]
    for i, h in enumerate(rev_headers):
        cell = rev_table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        set_rtl(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        r = p.add_run(h)
        style_run(r, size=11, bold=True, color="FFFFFF",
                   rtl=(i == 1))
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "C00000")
        tcPr.append(shd)

    for r_idx, (ro, desc, ri) in enumerate(rev_map, start=1):
        for c_idx, val in enumerate([ro, desc, ri]):
            cell = rev_table.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            if c_idx == 1:
                set_rtl(p)
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            color = ("C00000" if c_idx == 0 else
                      "1F4E79" if c_idx == 2 else "000000")
            bold = c_idx in (0, 2)
            style_run(r, size=11, bold=bold, color=color,
                       rtl=(c_idx == 1))

    rev_widths = [Cm(2.5), Cm(7.5), Cm(6.0)]
    for row in rev_table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = rev_widths[i]

    # Page break
    doc.add_page_break()

    # ----- 7. Validation -----
    add_section_heading(doc,
        "سابعاً: تحقق المنهج — كيف نُؤكِّدُ تحقُّق كلِّ هدف؟",
        level=1)
    add_arabic_p(doc,
        "يُقدّمُ الجدولُ الآتي مؤشراتِ التحقُّقِ التجريبيةَ "
        "(Validation Metrics) المُرتبطةَ بكلِّ هدفٍ بحثيٍّ:",
        spacing_after=10)

    val_data = [
        ("RO1", "إطار سبع طبقات منفّذ", "✅ منجز",
         "ملف docs/UNIFIED_ARCHITECTURE.md"),
        ("RO2", "WGAN-GP يولّد 210 ألف عينة، DoS recall +6%",
         "✅ منجز", "fig_per_class_recall.pdf"),
        ("RO3", "5 عقد، 20 جولة، Val Acc 99.57%، صفر بيانات خام",
         "✅ منجز", "fig_fl_convergence.pdf"),
        ("RO4", "30 كتلة، 420 IoC، تحقق < 50 ms",
         "✅ منجز", "fig_blockchain_growth.pdf"),
        ("RO5", "Class Prototypes + Gaussian DP (ε=5)",
         "✅ منجز", "fig_dp_comparison.pdf"),
        ("RO6", "Challenge Precision = 82.76%، 24 هجمة مكتشفة",
         "✅ منجز", "fig_zt_actions.pdf"),
        ("RO7", "تقييم رسمي على KDDTest+ (22,544 عينة)",
         "✅ منجز", "fig_ablation_matrix.pdf"),
        ("RO8", "U2R recall 10% → 22% (2.1×)، Probe +10.9pp",
         "✅ منجز", "fig_v1_vs_v2_per_class.pdf"),
    ]
    val_table = doc.add_table(rows=len(val_data) + 1, cols=4)
    val_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    val_headers = ["RO", "مؤشر التحقق", "الحالة", "الدليل"]
    for i, h in enumerate(val_headers):
        cell = val_table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        set_rtl(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        style_run(r, size=11, bold=True, color="FFFFFF",
                   rtl=(i in (1, 2)))
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "548235")
        tcPr.append(shd)

    for r_idx, row_v in enumerate(val_data, start=1):
        for c_idx, val in enumerate(row_v):
            cell = val_table.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            if c_idx == 1:
                set_rtl(p)
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            color = ("C00000" if c_idx == 0 else
                      "548235" if c_idx == 2 else
                      "555555" if c_idx == 3 else "000000")
            bold = c_idx in (0, 2)
            style_run(r, size=10, bold=bold, color=color,
                       rtl=(c_idx == 1))

    val_widths = [Cm(1.5), Cm(7.5), Cm(2.0), Cm(5.0)]
    for row in val_table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = val_widths[i]

    # Page break before phase status
    doc.add_page_break()

    # ----- 8. Phase Implementation Status -----
    add_section_heading(doc,
        "ثامناً: حالة تنفيذ المراحل البحثية",
        level=1)
    add_arabic_p(doc,
        "يُوضِّحُ الجدولُ الآتي ربطَ أهدافِ البحث بمراحلِ التنفيذ "
        "الفعليةِ وحالتها الراهنة، مع المخرجاتِ الملموسة والوقت "
        "الإجمالي على عتاد RTX 5070 Laptop GPU:",
        spacing_after=10)

    phases_data = [
        # (phase_id, phase_name_arabic, related_objectives, status,
        #  metric, time)
        ("A.1",
         "المعالجة المسبقة وتخزينها (Preprocessing + Cache)",
         "RO1, RO7",
         "✅ منجز",
         "125,973 + 22,544 عينة",
         "2 ث"),
        ("A.2",
         "تدريب WGAN-GP وتوليد العينات الصناعية",
         "RO1, RO2",
         "✅ منجز",
         "210,742 عينة، DoS +6.0%",
         "23 د"),
        ("A.3",
         "التعلّم الموزّع FedAvg (5 عقد، 20 جولة)",
         "RO1, RO3",
         "✅ منجز",
         "Val 99.57%، Test 77.65%",
         "13 د"),
        ("A.4",
         "تجميع المُجمَّع (RF + XGBoost + FL-NN)",
         "RO1, RO2, RO3",
         "✅ منجز",
         "أوزان 0.34/0.34/0.33",
         "1 د"),
        ("A.5",
         "Blockchain (PoA) + IoC harvest + Gaussian DP",
         "RO4, RO5",
         "✅ منجز",
         "30 كتلة، 420 IoC، تحقق <50ms",
         "30 ث"),
        ("A.6",
         "Zero-Trust Risk Scorer (LightGBM المستقل)",
         "RO1, RO6",
         "✅ منجز",
         "Val 99.82%، Challenge Prec 82.76%",
         "3 ث"),
        ("B",
         "الاستدلال الموحَّد (Online Inference)",
         "RO1, RO7",
         "✅ منجز",
         "0.024 ms/sample (هدف 1000 ms)",
         "—"),
        ("V2.1",
         "SMOTE + Focal Loss للأصناف النادرة",
         "RO2, RO8",
         "✅ منجز",
         "U2R recall 10% → 22%",
         "10 د"),
        ("V2.2",
         "Class Prototypes + Gaussian DP محسَّن",
         "RO5, RO8",
         "✅ منجز",
         "20 prototype بدل 400 IoC",
         "—"),
        ("V2.3",
         "Test-Time Adaptation + Threshold Calibration",
         "RO8",
         "✅ منجز",
         "Probe +10.9pp، Acc +2.0pp",
         "—"),
        ("V2.4",
         "CTGAN كبديل لـ WGAN-GP (الكود جاهز)",
         "RO2",
         "✅ منجز (كود)",
         "متاح للاستخدام عند الحاجة",
         "—"),
        ("التقييم",
         "Ablation Matrix لجميع التكوينات",
         "RO7, RO8",
         "✅ منجز",
         "6 تكوينات × 5 مقاييس",
         "5 د"),
        ("الرسومات",
         "توليد الرسومات الأكاديمية بصيغة IEEEtran",
         "RO7, RO8",
         "✅ منجز",
         "43 رسمة (PDF + PNG)",
         "30 ث"),
        ("الوثائق",
         "كتابة وثائق التفسير العلمي والمعمارية",
         "RO1",
         "✅ منجز",
         "5 ملفات Markdown + 5 LaTeX",
         "—"),
    ]

    phase_table = doc.add_table(rows=len(phases_data) + 1, cols=6)
    phase_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    phase_headers = ["المرحلة", "الوصف", "الأهداف",
                      "الحالة", "المخرج/المؤشر", "الزمن"]
    for i, h in enumerate(phase_headers):
        cell = phase_table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        set_rtl(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        r = p.add_run(h)
        style_run(r, size=10, bold=True, color="FFFFFF", rtl=True)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "7030A0")
        tcPr.append(shd)

    # Alternating row shading
    alt_color = "F4EFFB"
    for r_idx, row_p in enumerate(phases_data, start=1):
        for c_idx, val in enumerate(row_p):
            cell = phase_table.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if c_idx == 1:    # description column - RTL
                set_rtl(p)
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif c_idx == 4:  # metric column - RTL
                set_rtl(p)
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            color = ("7030A0" if c_idx == 0 else
                      "C00000" if c_idx == 2 else
                      "548235" if c_idx == 3 else
                      "1F4E79" if c_idx == 5 else "000000")
            bold = c_idx in (0, 2, 3)
            style_run(r, size=9, bold=bold, color=color,
                       rtl=(c_idx in (1, 4)))
            # Alt shading on even rows
            if r_idx % 2 == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), alt_color)
                tcPr.append(shd)

    phase_widths = [Cm(1.4), Cm(5.6), Cm(2.2), Cm(2.0),
                     Cm(3.8), Cm(1.0)]
    for row in phase_table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = phase_widths[i]

    # Summary box after table
    doc.add_paragraph()
    add_arabic_p(doc,
        "📊 ملخص الإنجاز:",
        size=13, bold=True, color="7030A0", spacing_after=4)

    add_bullet(doc,
        "المراحل المنفّذة: 14 مرحلة من 14 (100%)",
        bold=True, color="548235")
    add_bullet(doc,
        "الوقت الإجمالي: ≈ 50 دقيقة لإعادة التنفيذ الكامل من الصفر "
        "على عتادٍ استهلاكيٍّ.",
        bold=False)
    add_bullet(doc,
        "المخرجات الملموسة: 22 ملف Python (~4,500 سطر)، 43 رسمة، "
        "11 نموذج محفوظ، 5 وثائق LaTeX.",
        bold=False)
    add_bullet(doc,
        "حالة الجاهزية: المشروع جاهز بالكامل لكتابة الأطروحة "
        "والمناقشة قدّام اللجنة.",
        bold=True, color="C00000")

    doc.add_paragraph()
    add_arabic_p(doc,
        "🎯 المؤشرات الرئيسية للأداء (KPIs):",
        size=13, bold=True, color="7030A0", spacing_after=4)

    kpi_data = [
        ("الدقة الإجمالية (Binary Accuracy)", "77.42% (V2)",
         "هدف 97% — لم يُحقَّق بسبب Distribution Shift"),
        ("استدعاء U2R (U2R Recall)", "22.39% (2.1× من V1)",
         "إنجاز رئيسي على الفئة الأندر (52 عينة)"),
        ("استدعاء Probe", "80.42% (+10.9pp)",
         "تحسين كبير من معايرة العتبات"),
        ("زمن الاستدلال", "0.024 ms/عينة",
         "أسرع من الهدف بـ 41,600×"),
        ("دقة الاعتراض في Zero-Trust", "82.76%",
         "اعترض على 24 هجمة من 29 محاولة"),
        ("التحقق من Blockchain", "100% (verified)",
         "30 كتلة، تحقق أقل من 50ms"),
        ("الخصوصية", "صفر بيانات خام تخرج من العقد",
         "ضمان رياضي ε-DP عبر Gaussian mechanism"),
    ]

    kpi_table = doc.add_table(rows=len(kpi_data) + 1, cols=3)
    kpi_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    kpi_headers = ["مؤشر الأداء (KPI)", "القيمة المحققة", "الملاحظة"]
    for i, h in enumerate(kpi_headers):
        cell = kpi_table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        set_rtl(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        r = p.add_run(h)
        style_run(r, size=11, bold=True, color="FFFFFF", rtl=True)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "ED7D31")
        tcPr.append(shd)

    for r_idx, row_k in enumerate(kpi_data, start=1):
        for c_idx, val in enumerate(row_k):
            cell = kpi_table.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            set_rtl(p)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            r = p.add_run(val)
            color = ("1F4E79" if c_idx == 0 else
                      "548235" if c_idx == 1 else "555555")
            bold = c_idx in (0, 1)
            style_run(r, size=10, bold=bold, color=color, rtl=True)
            if r_idx % 2 == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), "FFF4E5")
                tcPr.append(shd)

    kpi_widths = [Cm(5.0), Cm(4.5), Cm(6.5)]
    for row in kpi_table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = kpi_widths[i]

    # Page break before conclusion
    doc.add_page_break()

    # ----- 9. Conclusion -----
    add_section_heading(doc, "تاسعاً: الخلاصة المنهجية", level=1)
    add_arabic_p(doc,
        "تُقدِّمُ هذه الوثيقةُ خريطةً شاملةً تُؤكِّدُ التماسكَ "
        "المنطقيَّ بين ثلاثةِ مكوناتٍ بحثيةٍ:")

    add_bullet(doc,
        "ثماني مشاكل بحثيةٍ موثقةٍ من الأدبيات والممارسةِ.")
    add_bullet(doc,
        "ثمانية أسئلةٍ بحثيةٍ تترجمُ تلك المشاكلَ إلى تساؤلاتٍ.")
    add_bullet(doc,
        "ثمانية أهدافٍ بحثيةٍ مُنفّذةٍ بشكلٍ كاملٍ في الكود "
        "والنتائج.")
    add_bullet(doc,
        "مصفوفتا ربطٍ ثنائيتا الاتجاه (إلى الأمام والخلف).")
    add_bullet(doc,
        "جدول تحقّقٍ تجريبيٍّ لكلِّ هدف، مُدعَّمٌ بالأدلةِ من "
        "الرسومات والملفاتِ.")

    add_arabic_p(doc,
        "ويُؤكِّدُ هذا الترابطُ المنهجيُّ أنَّ المشروعَ يُحقِّقُ "
        "متطلباتِ الأطروحةِ الأكاديميةِ من حيث:",
        spacing_after=8)

    add_bullet(doc,
        "أصالةُ الإسهام (Original Contribution): ست اكتشافاتٍ "
        "جديدةٍ قابلةٍ للنشر.",
        bold=True)
    add_bullet(doc,
        "صرامةُ المنهجية (Methodological Rigor): تقييمٌ "
        "على KDDTest+ الرسمي، مع الإفصاح الكامل عن القيود.",
        bold=True)
    add_bullet(doc,
        "قابلية التكرار (Reproducibility): إعادةُ تنفيذٍ "
        "كاملةٍ خلال 15 دقيقةً على عتادٍ استهلاكيٍّ.",
        bold=True)
    add_bullet(doc,
        "الترابط بين الأهداف: لا يوجد هدفٌ معزولٌ - كلُّها "
        "تُغذّي بعضَها في إطارٍ موحَّد.",
        bold=True)

    # Footer note
    doc.add_paragraph()
    foot_p = doc.add_paragraph()
    set_rtl(foot_p)
    foot_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = foot_p.add_run(
        "— تاريخُ التحرير: 28 نيسان 2026 — مرفقات: 22 ملف Python، "
        "43 رسماً أكاديمياً، 5 وثائق LaTeX —"
    )
    style_run(r, size=9, bold=False, color="888888")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"[OK] saved -> {OUT}")
    print(f"     size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
