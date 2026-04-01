"""
PDF Report Generator for Tremayne Performance v5.

Generates professional athlete performance reports from DFA A1 ramp test
analysis results, including threshold data, metabolic profiling, data
quality metrics, historical comparison, and coach comments.
"""

import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage
)


# ---------------------------------------------------------------------------
# COLOUR PALETTE
# ---------------------------------------------------------------------------
DARK_BG = colors.HexColor('#0e0f12')
ACCENT_BLUE = colors.HexColor('#2762d6')
LIGHT_BLUE = colors.HexColor('#60a5fa')
ACCENT_GREEN = colors.HexColor('#22c55e')
ACCENT_RED = colors.HexColor('#ef4444')
ACCENT_ORANGE = colors.HexColor('#f97316')
TEXT_PRIMARY = colors.HexColor('#1f2937')
TEXT_SECONDARY = colors.HexColor('#6b7280')
TEXT_LIGHT = colors.HexColor('#9ca3af')
TABLE_HEADER_BG = colors.HexColor('#1e293b')
TABLE_ROW_ALT = colors.HexColor('#f8fafc')
TABLE_BORDER = colors.HexColor('#e2e8f0')
POSITIVE_GREEN = colors.HexColor('#059669')
NEGATIVE_RED = colors.HexColor('#dc2626')


# ---------------------------------------------------------------------------
# CUSTOM STYLES
# ---------------------------------------------------------------------------
def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'ReportTitle', parent=styles['Title'],
        fontName='Helvetica-Bold', fontSize=18, leading=22,
        textColor=TEXT_PRIMARY, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        'ReportSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=13,
        textColor=TEXT_SECONDARY, spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=12, leading=15,
        textColor=ACCENT_BLUE, spaceBefore=6 * mm, spaceAfter=3 * mm,
        borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        'BodyText2', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=12,
        textColor=TEXT_PRIMARY,
    ))
    styles.add(ParagraphStyle(
        'SmallNote', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=TEXT_LIGHT,
    ))
    styles.add(ParagraphStyle(
        'CoachComment', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=13,
        textColor=TEXT_PRIMARY, leftIndent=4 * mm,
        borderWidth=1, borderColor=ACCENT_BLUE, borderPadding=3 * mm,
    ))
    styles.add(ParagraphStyle(
        'DeltaPositive', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9, textColor=POSITIVE_GREEN,
    ))
    styles.add(ParagraphStyle(
        'DeltaNegative', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9, textColor=NEGATIVE_RED,
    ))
    styles.add(ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7, leading=9,
        textColor=TEXT_LIGHT, alignment=TA_CENTER,
    ))
    return styles


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def _safe(val, fmt='{:.0f}', suffix='', fallback='—'):
    """Format a value safely, returning fallback if None."""
    if val is None:
        return fallback
    try:
        return fmt.format(float(val)) + suffix
    except (ValueError, TypeError):
        return str(val) + suffix


def _delta_str(current, previous, fmt='{:+.0f}', suffix='', pct=True):
    """Format a delta between current and previous values."""
    if current is None or previous is None:
        return '—'
    try:
        c, p = float(current), float(previous)
        d = c - p
        result = fmt.format(d) + suffix
        if pct and p != 0:
            pct_val = (d / p) * 100
            result += f' ({pct_val:+.1f}%)'
        return result
    except (ValueError, TypeError):
        return '—'


def _pace_str(speed_ms):
    """Convert speed (m/s) to pace string (M:SS/km)."""
    if not speed_ms or speed_ms <= 0:
        return '—'
    pace_sec = 1000.0 / speed_ms
    m = int(pace_sec // 60)
    s = int(pace_sec % 60)
    return f'{m}:{s:02d}/km'


def _divider():
    return HRFlowable(
        width='100%', thickness=0.5, color=TABLE_BORDER,
        spaceBefore=2 * mm, spaceAfter=2 * mm
    )


# ---------------------------------------------------------------------------
# TABLE BUILDERS
# ---------------------------------------------------------------------------
def _threshold_table(thr, is_run, weight_kg, styles):
    """Build a 3-column threshold comparison table."""
    if is_run:
        headers = ['Metric', 'HRVT1 Standard', 'HRVT1 Individualised', 'HRVT2 / Threshold']
        data = [
            headers,
            ['Pace', _pace_str(thr.get('hrvt1s_power')),
             _pace_str(thr.get('hrvt1c_power')),
             _pace_str(thr.get('hrvt2_power'))],
            ['Speed (m/s)', _safe(thr.get('hrvt1s_power'), '{:.2f}'),
             _safe(thr.get('hrvt1c_power'), '{:.2f}'),
             _safe(thr.get('hrvt2_power'), '{:.2f}')],
            ['Heart Rate', _safe(thr.get('hrvt1s_hr'), suffix=' bpm'),
             _safe(thr.get('hrvt1c_hr'), suffix=' bpm'),
             _safe(thr.get('hrvt2_hr'), suffix=' bpm')],
            ['α1 Cutoff', '0.75',
             _safe(thr.get('a1_star'), '{:.2f}'),
             '0.50'],
        ]
    else:
        headers = ['Metric', 'HRVT1 Standard', 'HRVT1 Individualised', 'HRVT2 / FTP']
        wkg_row = None
        if weight_kg and weight_kg > 0:
            wkg_row = ['W/kg',
                       _safe(thr.get('hrvt1s_power'), '{:.2f}', suffix='') if thr.get('hrvt1s_power') else '—',
                       _safe(thr.get('hrvt1c_power'), '{:.2f}', suffix='') if thr.get('hrvt1c_power') else '—',
                       _safe(thr.get('hrvt2_power'), '{:.2f}', suffix='') if thr.get('hrvt2_power') else '—']
            # Calculate w/kg
            for i in range(1, 4):
                key = ['hrvt1s_power', 'hrvt1c_power', 'hrvt2_power'][i - 1]
                v = thr.get(key)
                if v and weight_kg:
                    wkg_row[i] = f'{float(v) / weight_kg:.2f}'

        data = [
            headers,
            ['Power (W)', _safe(thr.get('hrvt1s_power'), suffix='W'),
             _safe(thr.get('hrvt1c_power'), suffix='W'),
             _safe(thr.get('hrvt2_power'), suffix='W')],
            ['Heart Rate', _safe(thr.get('hrvt1s_hr'), suffix=' bpm'),
             _safe(thr.get('hrvt1c_hr'), suffix=' bpm'),
             _safe(thr.get('hrvt2_hr'), suffix=' bpm')],
            ['α1 Cutoff', '0.75',
             _safe(thr.get('a1_star'), '{:.2f}'),
             '0.50'],
        ]
        if wkg_row:
            data.insert(2, wkg_row)

    col_widths = [28 * mm, 38 * mm, 42 * mm, 38 * mm]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (0, -1), TEXT_SECONDARY),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, TABLE_ROW_ALT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def _quality_table(data, styles):
    """Build a data quality summary table."""
    rv = data.get('ramp_validation', {})
    ev = data.get('effort_validation', {})
    dq = data.get('data_quality', {})
    reg = data.get('regression_power', {})

    rows = [
        ['Metric', 'Value'],
        ['Stages Completed', f"{rv.get('stages_completed', '?')} / 10"],
        ['Ramp Status', rv.get('overall_status', '?')],
        ['Effort Status', ev.get('status', '?')],
        ['R² (Power)', _safe(reg.get('r2'), '{:.3f}')],
        ['Artifact Rate', _safe(dq.get('artifact_rate_ramp'), '{:.1f}', suffix='%')],
        ['Data Quality', _safe(dq.get('overall_quality'), fallback='?')],
    ]

    t = Table(rows, colWidths=[40 * mm, 40 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (0, -1), TEXT_SECONDARY),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, TABLE_ROW_ALT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def _comparison_table(current, previous, is_run, styles):
    """Build a comparison table between current and previous test."""
    if is_run:
        rows = [
            ['Metric', 'Previous', 'Current', 'Change'],
            ['HRVT2 Pace', _pace_str(previous.get('hrvt2_power')),
             _pace_str(current.get('hrvt2_power')),
             _delta_str(current.get('hrvt2_power'), previous.get('hrvt2_power'),
                        fmt='{:+.2f}', suffix=' m/s')],
            ['HRVT1c Pace', _pace_str(previous.get('hrvt1c_power')),
             _pace_str(current.get('hrvt1c_power')),
             _delta_str(current.get('hrvt1c_power'), previous.get('hrvt1c_power'),
                        fmt='{:+.2f}', suffix=' m/s')],
            ['HRVT2 HR', _safe(previous.get('hrvt2_hr'), suffix=' bpm'),
             _safe(current.get('hrvt2_hr'), suffix=' bpm'),
             _delta_str(current.get('hrvt2_hr'), previous.get('hrvt2_hr'), suffix=' bpm', pct=False)],
            ['Archetype', previous.get('archetype', '—'),
             current.get('archetype', '—'), '—'],
            ['R²', _safe(previous.get('regression_r2_power'), '{:.3f}'),
             _safe(current.get('regression_r2_power'), '{:.3f}'), '—'],
        ]
    else:
        rows = [
            ['Metric', 'Previous', 'Current', 'Change'],
            ['HRVT2 / FTP', _safe(previous.get('hrvt2_power'), suffix='W'),
             _safe(current.get('hrvt2_power'), suffix='W'),
             _delta_str(current.get('hrvt2_power'), previous.get('hrvt2_power'), suffix='W')],
            ['HRVT1c', _safe(previous.get('hrvt1c_power'), suffix='W'),
             _safe(current.get('hrvt1c_power'), suffix='W'),
             _delta_str(current.get('hrvt1c_power'), previous.get('hrvt1c_power'), suffix='W')],
            ['HRVT2 HR', _safe(previous.get('hrvt2_hr'), suffix=' bpm'),
             _safe(current.get('hrvt2_hr'), suffix=' bpm'),
             _delta_str(current.get('hrvt2_hr'), previous.get('hrvt2_hr'), suffix=' bpm', pct=False)],
            ['Archetype', previous.get('archetype', '—'),
             current.get('archetype', '—'), '—'],
            ['R²', _safe(previous.get('regression_r2_power'), '{:.3f}'),
             _safe(current.get('regression_r2_power'), '{:.3f}'), '—'],
        ]

    col_widths = [30 * mm, 35 * mm, 35 * mm, 46 * mm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (0, -1), TEXT_SECONDARY),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, TABLE_ROW_ALT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


# ---------------------------------------------------------------------------
# COMPARISON SUMMARY GENERATOR
# ---------------------------------------------------------------------------
def _generate_comparison_summary(current_thr, previous, is_run):
    """Generate a plain-text summary of how performance has changed."""
    sentences = []

    hrvt2_cur = current_thr.get('hrvt2_power')
    hrvt2_prev = previous.get('hrvt2_power')
    hrvt1c_cur = current_thr.get('hrvt1c_power')
    hrvt1c_prev = previous.get('hrvt1c_power')

    if hrvt2_cur is not None and hrvt2_prev is not None:
        d = float(hrvt2_cur) - float(hrvt2_prev)
        pct = (d / float(hrvt2_prev)) * 100 if float(hrvt2_prev) != 0 else 0
        if is_run:
            if d > 0:
                sentences.append(f'Threshold speed increased {pct:.1f}% ({d:+.2f} m/s).')
            elif d < 0:
                sentences.append(f'Threshold speed decreased {abs(pct):.1f}% ({d:+.2f} m/s).')
            else:
                sentences.append('Threshold speed unchanged.')
        else:
            if d > 0:
                sentences.append(f'FTP increased {pct:.1f}% ({d:+.0f}W).')
            elif d < 0:
                sentences.append(f'FTP decreased {abs(pct):.1f}% ({d:+.0f}W).')
            else:
                sentences.append('FTP unchanged.')

    if hrvt1c_cur is not None and hrvt1c_prev is not None:
        d = float(hrvt1c_cur) - float(hrvt1c_prev)
        if is_run:
            if d > 0:
                sentences.append(f'Aerobic threshold speed improved ({d:+.2f} m/s).')
            elif d < 0:
                sentences.append(f'Aerobic threshold speed declined ({d:+.2f} m/s).')
        else:
            if d > 0:
                sentences.append(f'Aerobic threshold improved ({d:+.0f}W).')
            elif d < 0:
                sentences.append(f'Aerobic threshold declined ({d:+.0f}W).')

    arch_cur = current_thr.get('archetype') if isinstance(current_thr, dict) else None
    arch_prev = previous.get('archetype')
    if arch_cur and arch_prev and arch_cur != arch_prev:
        sentences.append(f'Archetype changed from {arch_prev} to {arch_cur}.')

    hr_cur = current_thr.get('hrvt2_hr')
    hr_prev = previous.get('hrvt2_hr')
    if hr_cur is not None and hr_prev is not None:
        d = int(hr_cur) - int(hr_prev)
        if d != 0:
            direction = 'increased' if d > 0 else 'decreased'
            sentences.append(f'HRVT2 heart rate {direction} {abs(d)} bpm.')

    if not sentences:
        sentences.append('Insufficient data for detailed comparison.')

    return ' '.join(sentences)


# ---------------------------------------------------------------------------
# MAIN PDF GENERATOR
# ---------------------------------------------------------------------------
def generate_ramp_report(result_data, athlete_name, sport, coach_comments='',
                         previous_test=None, weight_kg=None, hrmax=None):
    """
    Generate a PDF report for a DFA A1 ramp test.

    Args:
        result_data: Full analysis result dict from /analyze_ramp
        athlete_name: Athlete's name
        sport: 'bike' or 'run'
        coach_comments: Free-text coach notes
        previous_test: Previous test history record (for comparison), or None
        weight_kg: Athlete body weight in kg
        hrmax: HRmax for the sport

    Returns:
        bytes: PDF file contents
    """
    buf = io.BytesIO()
    styles = _build_styles()
    is_run = (sport == 'run' or result_data.get('protocol_type') == 'run')

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title=f'Performance Report — {athlete_name}',
        author='Tremayne Performance v5',
    )

    story = []
    thr = result_data.get('thresholds', {})
    arch = result_data.get('archetype', {})
    now = datetime.now()

    # ── HEADER ──────────────────────────────────────────────────────
    story.append(Paragraph('TREMAYNE PERFORMANCE', styles['ReportTitle']))
    sport_label = 'Running' if is_run else 'Cycling'
    story.append(Paragraph(
        f'DFA α1 Ramp Test Report &nbsp;|&nbsp; {sport_label} &nbsp;|&nbsp; '
        f'{now.strftime("%d %B %Y")}',
        styles['ReportSubtitle']
    ))

    # Athlete info line
    info_parts = [f'<b>Athlete:</b> {athlete_name}']
    if weight_kg:
        info_parts.append(f'<b>Weight:</b> {weight_kg} kg')
    if hrmax:
        info_parts.append(f'<b>HRmax:</b> {hrmax} bpm')
    story.append(Paragraph(' &nbsp;|&nbsp; '.join(info_parts), styles['BodyText2']))
    story.append(_divider())

    # ── THRESHOLDS ──────────────────────────────────────────────────
    story.append(Paragraph('THRESHOLDS', styles['SectionHeading']))

    extrapolated = thr.get('hrvt2_extrapolated', False)
    if extrapolated:
        story.append(Paragraph(
            '<i>Note: HRVT2 was extrapolated beyond the observed data range.</i>',
            styles['SmallNote']
        ))

    story.append(_threshold_table(thr, is_run, weight_kg, styles))
    story.append(Spacer(1, 2 * mm))

    # CI note
    ci = thr.get('hrvt2_ci_95')
    if ci and not is_run:
        story.append(Paragraph(
            f'HRVT2 95% Confidence Interval: {ci[0]}–{ci[1]}W',
            styles['SmallNote']
        ))

    # ── THRESHOLD CHART ────────────────────────────────────────────
    try:
        import chart_renderer
        chart_png = chart_renderer.render_threshold_chart(
            stage_data=result_data.get('stage_data', []),
            regression=result_data.get('regression_power', {}),
            thresholds=thr,
            windows=result_data.get('windows', []),
            effort_validation=result_data.get('effort_validation'),
            is_run=is_run,
        )
        if chart_png:
            story.append(Spacer(1, 3 * mm))
            chart_img = RLImage(io.BytesIO(chart_png), width=170 * mm, height=85 * mm)
            story.append(chart_img)
            story.append(Spacer(1, 2 * mm))
    except Exception:
        # Chart generation failed — continue with text-only report
        pass

    # ── METABOLIC PROFILE ───────────────────────────────────────────
    if arch and arch.get('archetype'):
        story.append(Paragraph('METABOLIC PROFILE', styles['SectionHeading']))

        conf = arch.get('confidence', '')
        conf_label = ('Full Confidence' if conf == 'high' else
                      'Reduced Confidence' if conf == 'medium' else
                      'Threshold Ratios Only')

        profile_data = [
            ['Classification', arch.get('archetype', '—')],
            ['Confidence', conf_label],
        ]
        if arch.get('ceiling_limited'):
            profile_data.append(['Note', 'Ceiling-Limited (power capped)'])

        dev = arch.get('development_level')
        if dev:
            profile_data.append(['Development', f"{dev.get('level', '—')} — {dev.get('note', '')}"])

        t = Table(profile_data, colWidths=[35 * mm, 110 * mm])
        t.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), TEXT_SECONDARY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t)

        # Strengths & Weaknesses
        fb = arch.get('feedback', {})
        strengths = fb.get('strengths', [])
        weaknesses = fb.get('weaknesses', [])

        if strengths or weaknesses:
            story.append(Spacer(1, 2 * mm))
            sw_data = [['Strengths', 'Weaknesses']]
            max_len = max(len(strengths), len(weaknesses), 1)
            for i in range(max_len):
                s = strengths[i] if i < len(strengths) else ''
                w = weaknesses[i] if i < len(weaknesses) else ''
                sw_data.append([s, w])

            sw_table = Table(sw_data, colWidths=[73 * mm, 73 * mm])
            sw_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), ACCENT_GREEN),
                ('BACKGROUND', (1, 0), (1, 0), ACCENT_RED),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, TABLE_BORDER),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, TABLE_ROW_ALT]),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            story.append(sw_table)

        # Training recommendations
        recs = arch.get('training_recommendations', [])
        if recs:
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph('<b>Training Recommendations</b>', styles['BodyText2']))
            for r in recs:
                story.append(Paragraph(f'• {r}', styles['BodyText2']))

    # ── DATA QUALITY ────────────────────────────────────────────────
    story.append(Paragraph('DATA QUALITY', styles['SectionHeading']))
    story.append(_quality_table(result_data, styles))

    # Warnings
    warnings = result_data.get('warnings', [])
    if warnings:
        story.append(Spacer(1, 2 * mm))
        for w in warnings[:5]:  # limit to 5
            story.append(Paragraph(f'⚠ {w}', styles['SmallNote']))

    # ── HISTORICAL COMPARISON ───────────────────────────────────────
    if previous_test:
        story.append(Paragraph('COMPARISON TO PREVIOUS TEST', styles['SectionHeading']))

        prev_date = previous_test.get('test_date', '')
        if prev_date:
            prev_date_str = prev_date.split('T')[0] if 'T' in prev_date else prev_date
            story.append(Paragraph(
                f'Previous test date: <b>{prev_date_str}</b>',
                styles['BodyText2']
            ))
            story.append(Spacer(1, 2 * mm))

        # Build current thresholds dict matching history format for comparison
        current_for_compare = {
            'hrvt2_power': thr.get('hrvt2_power'),
            'hrvt1c_power': thr.get('hrvt1c_power'),
            'hrvt2_hr': thr.get('hrvt2_hr'),
            'hrvt1c_hr': thr.get('hrvt1c_hr'),
            'archetype': arch.get('archetype') if arch else None,
            'regression_r2_power': (result_data.get('regression_power', {}) or {}).get('r2'),
        }

        story.append(_comparison_table(current_for_compare, previous_test, is_run, styles))
        story.append(Spacer(1, 3 * mm))

        # Auto-generated summary
        summary = _generate_comparison_summary(current_for_compare, previous_test, is_run)
        story.append(Paragraph(f'<b>Summary:</b> {summary}', styles['BodyText2']))

    # ── COACH COMMENTS ──────────────────────────────────────────────
    if coach_comments and coach_comments.strip():
        story.append(Paragraph('COACH COMMENTS', styles['SectionHeading']))
        # Split by newlines and render each line
        for line in coach_comments.strip().split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), styles['CoachComment']))
            else:
                story.append(Spacer(1, 2 * mm))

    # ── FOOTER ──────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(_divider())
    story.append(Paragraph(
        f'Generated by Tremayne Performance v5 &nbsp;|&nbsp; '
        f'{now.strftime("%d/%m/%Y %H:%M")} &nbsp;|&nbsp; '
        f'DFA α1 analysis using Kubios-aligned preprocessing pipeline',
        styles['Footer']
    ))

    # Build PDF
    doc.build(story)
    return buf.getvalue()
