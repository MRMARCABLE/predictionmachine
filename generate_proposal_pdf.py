import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Kanvas kustom dua putaran untuk penomoran 'Halaman X dari Y' dan running header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Oblique", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Running header (mulai halaman 2 ke atas)
        if self._pageNumber > 1:
            self.drawString(54, 802, "PROPOSAL PROYEK P08 — PREDIKSI IPK & PERBANDINGAN ALGORITMA")
            self.setStrokeColor(colors.HexColor("#D0D5DD"))
            self.setLineWidth(0.6)
            self.line(54, 795, 541, 795)
        
        # Running footer
        page_text = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawRightString(541, 32, page_text)
        self.drawString(54, 32, "Kelompok 1 · Decision Tree vs Random Forest · Explainable AI (SHAP)")
        self.setStrokeColor(colors.HexColor("#D0D5DD"))
        self.setLineWidth(0.6)
        self.line(54, 44, 541, 44)
        self.restoreState()


def create_proposal_pdf(output_filename: str = "PROPOSAL_BAB_1_KELOMPOK_1.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14.5,
        leading=18.5,
        alignment=1,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14.5,
        alignment=1,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=12,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        alignment=1,
        textColor=colors.HexColor("#4A5568"),
    )

    chapter_title_style = ParagraphStyle(
        "ChapterTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#1A202C"),
        spaceBefore=10,
        spaceAfter=12,
    )

    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        alignment=4,
        textColor=colors.HexColor("#2D3748"),
        firstLineIndent=18,
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        alignment=4,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3.5,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=colors.white,
    )

    table_body_style = ParagraphStyle(
        "TableBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        alignment=0,
        textColor=colors.HexColor("#2D3748"),
    )

    box_text_style = ParagraphStyle(
        "BoxText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1A365D"),
    )

    story = []

    # =========================================================================
    # KOP & JUDUL PROPOSAL (SESUAI SPESIFIKASI TUGAS)
    # =========================================================================
    story.append(Paragraph("PROPOSAL PROYEK MACHINE LEARNING (P08)", subtitle_style))
    story.append(Paragraph(
        "SISTEM PREDIKSI INDEKS PRESTASI KUMULATIF (IPK) AKHIR SEMESTER DARI DATA AWAL SEMESTER DILENGKAPI EXPLAINABLE AI (SHAP) SEBAGAI SISTEM PENDUKUNG KEPUTUSAN DOSEN PEMBIMBING AKADEMIK",
        title_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceBefore=3, spaceAfter=6))
    story.append(Paragraph("<b>Disusun Oleh:</b> Kelompok 1 · <b>Stack:</b> Python, Scikit-learn, SHAP, Pandas, Streamlit", meta_style))
    story.append(Paragraph("<b>Komparasi Utama:</b> Decision Tree vs. Random Forest · <b>Metrik:</b> RMSE, MAE, R² · <b>Evaluasi Proyek:</b> ★★★☆☆", meta_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceBefore=6, spaceAfter=10))

    # =========================================================================
    # KOTAK RINGKASAN SPESIFIKASI TUGAS
    # =========================================================================
    spec_summary = [
        [
            Paragraph("<b>Ringkasan Spesifikasi & Ruang Lingkup Tugas:</b><br/>"
                      "• <b>Deskripsi:</b> Memprediksi IPK akhir semester dari data awal semester + visualisasi faktor berpengaruh (SHAP).<br/>"
                      "• <b>Dataset Terpilih:</b> UCI <i>Students' Dropout and Academic Success</i> (4.424 data) — dataset publik, bersih, tanpa data sensitif.<br/>"
                      "• <b>Scope Pemodelan:</b> Perbandingan dua algoritma (Decision Tree vs. Random Forest), evaluasi RMSE, MAE, R².<br/>"
                      "• <b>Antarmuka & Etika:</b> Input data 1 mahasiswa → Prediksi IPK + Bar Plot SHAP + Diskusi implikasi etis di pendidikan.", box_text_style)
        ]
    ]
    t_box = Table(spec_summary, colWidths=[17.0 * cm])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_box)
    story.append(Spacer(1, 8))

    # =========================================================================
    # BAB 1 PENDAHULUAN
    # =========================================================================
    story.append(Paragraph("BAB 1<br/>PENDAHULUAN", chapter_title_style))

    # --- 1.1 LATAR BELAKANG ---
    story.append(Paragraph("1.1 Latar Belakang Masalah", section_title_style))
    story.append(Paragraph(
        "Dalam ekosistem perguruan tinggi, keberhasilan akademik mahasiswa pada tahun pertama merupakan indikator vital yang menentukan retensi dan kelulusan tepat waktu. Namun, evaluasi performa akademik yang berjalan saat ini umumnya masih bersifat reaktif (<i>post-facto</i>). Dosen Pembimbing Akademik (PA) baru mengetahui penurunan performa setelah nilai akhir semester diumumkan secara resmi atau setelah mahasiswa mengalami akumulasi Indeks Prestasi Kumulatif (IPK) rendah. Kondisi ini menyebabkan bimbingan akademik terlambat diberikan ketika mahasiswa telah kehilangan motivasi atau terancam sanksi putus studi.",
        body_style
    ))
    story.append(Paragraph(
        "Untuk menjawab tantangan tersebut, penelitian ini mengembangkan sistem yang mampu memprediksi capaian IPK akhir semester sejak data awal semester tersedia. Dengan memanfaatkan dataset publik <i>Students' Dropout and Academic Success</i> (4.424 data) dari UCI Machine Learning Repository, model dilatih menggunakan fitur-fitur pendaftaran, demografi, sosioekonomi, serta capaian Semester 1 tanpa melibatkan data sensitif internal mahasiswa.",
        body_style
    ))
    story.append(Paragraph(
        "Fokus utama penelitian diarahkan pada perbandingan dua algoritma pembelajaran mesin berbasis pohon: <b>Decision Tree</b> dan <b>Random Forest</b>. Selain membandingkan akurasi prediksi menggunakan metrik standar regresi (RMSE, MAE, dan R²), sistem ini mengintegrasikan metode <b>Explainable AI (SHAP)</b> untuk menyajikan visualisasi <i>bar plot</i> faktor penentu prediksi. Melalui keterbukaan informasi ini, Dosen PA tidak hanya menerima angka estimasi IPK, tetapi juga memahami faktor pendorong (seperti beasiswa dan nilai masuk) serta faktor risiko (seperti tunggakan SPP) secara transparan, disertai pemahaman etika bahwa sistem bertindak sebagai alat bantu bimbingan, bukan vonis mutlak.",
        body_style
    ))

    # --- 1.2 RUMUSAN MASALAH ---
    story.append(Paragraph("1.2 Rumusan Masalah", section_title_style))
    story.append(Paragraph("Berdasarkan deskripsi dan batasan tugas di atas, rumusan masalah disusun sebagai berikut:", body_style))
    story.append(Paragraph("1. Bagaimana membangun model regresi berbasis <b>Decision Tree</b> dan <b>Random Forest</b> untuk memprediksi IPK akhir semester dari data awal semester bebas kebocoran target?", bullet_style))
    story.append(Paragraph("2. Bagaimana perbandingan performa akurasi antara algoritma <b>Decision Tree vs. Random Forest</b> berdasarkan metrik evaluasi RMSE, MAE, dan R²?", bullet_style))
    story.append(Paragraph("3. Bagaimana mengidentifikasi dan memvisualisasikan faktor paling berpengaruh terhadap prediksi IPK mahasiswa menggunakan <b>SHAP bar plot</b>?", bullet_style))
    story.append(Paragraph("4. Bagaimana merancang antarmuka aplikasi interaktif berbasis <b>Streamlit</b> yang menerima input data satu mahasiswa dan mendiskusikan implikasi etis sistem prediksi di lingkungan pendidikan?", bullet_style))

    # --- 1.3 BATASAN MASALAH & SCOPE ---
    story.append(Paragraph("1.3 Batasan Masalah (Scope & Pembatasan)", section_title_style))
    story.append(Paragraph("Ruang lingkup penelitian dibatasi secara ketat sesuai spesifikasi tugas:", body_style))
    story.append(Paragraph("1. <b>Dataset:</b> Menggunakan dataset publik UCI <i>Students' Dropout and Academic Success</i> berjumlah 4.424 baris data yang bersih dan tanpa data sensitif internal.", bullet_style))
    story.append(Paragraph("2. <b>Dua Model yang Dibandingkan:</b> Pembahasan komparatif difokuskan murni pada <b>Decision Tree Regressor</b> vs. <b>Random Forest Regressor</b>.", bullet_style))
    story.append(Paragraph("3. <b>Metrik Evaluasi:</b> Menggunakan tiga metrik regresi wajib, yaitu <i>Root Mean Squared Error</i> (RMSE), <i>Mean Absolute Error</i> (MAE), dan koefisien determinasi (R²).", bullet_style))
    story.append(Paragraph("4. <b>Feature Importance:</b> Analisis faktor paling berpengaruh disajikan secara visual menggunakan <i>horizontal bar plot</i> nilai atribusi SHAP.", bullet_style))
    story.append(Paragraph("5. <b>Antarmuka Pengguna:</b> Aplikasi Streamlit menerima input data satu mahasiswa dan menampilkan hasil prediksi IPK beserta rincian faktor penjelasnya.", bullet_style))

    # --- 1.4 TUJUAN PENELITIAN ---
    story.append(Paragraph("1.4 Tujuan Penelitian", section_title_style))
    story.append(Paragraph("Tujuan pelaksanaan proyek penelitian ini adalah:", body_style))
    story.append(Paragraph("1. Mengimplementasikan pipeline prediksi IPK akhir semester berbasis Python, Pandas, dan Scikit-learn yang bebas dari kebocoran target.", bullet_style))
    story.append(Paragraph("2. Menyajikan studi perbandingan empiris performa antara Decision Tree dan Random Forest dalam mengestimasi capaian akademik mahasiswa.", bullet_style))
    story.append(Paragraph("3. Menghasilkan visualisasi faktor dominan penentu prediksi menggunakan SHAP untuk meningkatkan interpretabilitas model bagi Dosen PA.", bullet_style))
    story.append(Paragraph("4. Membangun prototipe antarmuka Streamlit yang responsif dan etis sebagai sistem pendukung keputusan bimbingan akademik.", bullet_style))

    # --- 1.5 KONTRIBUSI ILMIAH & MANFAAT ---
    story.append(Paragraph("1.5 Kontribusi Ilmiah dan Implikasi Etis", section_title_style))
    story.append(Paragraph(
        "Penelitian ini memberikan tiga kontribusi ilmiah utama sesuai dengan sasaran pembelajaran:",
        body_style
    ))
    story.append(Paragraph("1. <b>Perbandingan Decision Tree vs. Random Forest:</b> Menyajikan analisis mendalam mengenai karakteristik model pohon tunggal (yang rentan <i>overfitting</i> tanpa pembatasan) dibandingkan dengan model ansambel Random Forest yang terbukti lebih stabil dan memiliki generalisasi lebih baik pada data akademik.", bullet_style))
    story.append(Paragraph("2. <b>Analisis SHAP untuk Faktor Penentu:</b> Membongkar sifat kotak hitam (<i>black-box</i>) model melalui SHAP <i>TreeExplainer</i>, sehingga mampu mengidentifikasi secara pasti faktor pendorong dan penghambat performa mahasiswa (seperti pengaruh rasio kelulusan SKS dan kelancaran pembayaran SPP).", bullet_style))
    story.append(Paragraph("3. <b>Diskusi Implikasi Etis Sistem Prediksi di Pendidikan:</b> Menegaskan batas etis bahwa sistem AI di dunia pendidikan berfungsi murni sebagai <i>Decision Support System</i> (alat bantu deteksi dini bagi Dosen PA) dan bukan merupakan vonis kelulusan otomatis, guna mencegah bias diskriminatif terhadap mahasiswa.", bullet_style))

    # --- 1.6 JUSTIFIKASI KOMPLEKSITAS ---
    story.append(Paragraph("1.6 Justifikasi Tingkat Kesulitan Proyek (★★★☆☆)", section_title_style))
    story.append(Paragraph(
        "Proyek ini diklasifikasikan pada tingkat kesulitan menengah (3 dari 5 bintang) dengan pertimbangan objektif sebagai berikut:",
        body_style
    ))
    story.append(Paragraph("• <b>Dataset Publik dan Bersih:</b> Data UCI (4.424 baris) telah terdokumentasi dengan baik, tidak memiliki nilai hilang (<i>missing values = 0</i>), dan bebas dari isu privasi sensitif.", bullet_style))
    story.append(Paragraph("• <b>Implementasi Model yang Terstruktur:</b> Pembangunan pipeline Decision Tree dan Random Forest didukung pustaka Scikit-learn yang matang dan teruji.", bullet_style))
    story.append(Paragraph("• <b>Dukungan Pustaka SHAP yang Solid:</b> Integrasi SHAP <i>TreeExplainer</i> pada model berbasis pohon memiliki dokumentasi resmi yang sangat lengkap dan efisien secara komputasi.", bullet_style))
    story.append(Paragraph("• <b>Pengembangan Antarmuka Cepat:</b> Pustaka Streamlit memungkinkan perancangan form input interaktif satu mahasiswa dan grafik visualisasi secara terintegrasi.", bullet_style))

    # --- 1.7 SISTEMATIKA PENULISAN ---
    story.append(Paragraph("1.7 Sistematika Penulisan Proposal", section_title_style))
    story.append(Paragraph("Sistematika pembahasan proposal proyek ini dirancang sebagai berikut:", body_style))

    table_data = [
        [
            Paragraph("<b>Struktur Bab</b>", table_header_style),
            Paragraph("<b>Fokus Pembahasan (Sesuai Spesifikasi Tugas)</b>", table_header_style),
        ],
        [
            Paragraph("<b>BAB 1 PENDAHULUAN</b>", table_body_style),
            Paragraph("Latar belakang intervensi dini Dosen PA, rumusan masalah, batasan Decision Tree vs Random Forest, tujuan, kontribusi ilmiah, dan justifikasi kompleksitas.", table_body_style),
        ],
        [
            Paragraph("<b>BAB 2 DASAR TEORI</b>", table_body_style),
            Paragraph("Teori regresi performa akademik, algoritma Decision Tree dan Random Forest, metrik RMSE/MAE/R², dan konsep nilai Shapley pada Explainable AI (SHAP).", table_body_style),
        ],
        [
            Paragraph("<b>BAB 3 METODOLOGI & DATA</b>", table_body_style),
            Paragraph("Eksplorasi dataset UCI (4.424 data), penghapusan target leakage, penyiapan 31 fitur input, dan partisi data train-test 80/20.", table_body_style),
        ],
        [
            Paragraph("<b>BAB 4 EKSPERIMEN & SHAP</b>", table_body_style),
            Paragraph("Pelatihan dan perbandingan Decision Tree vs Random Forest (5-Fold CV), evaluasi metrik, visualisasi SHAP bar plot faktor paling berpengaruh, dan diskusi etika.", table_body_style),
        ],
        [
            Paragraph("<b>BAB 5 APLIKASI STREAMLIT</b>", table_body_style),
            Paragraph("Perancangan antarmuka Streamlit: form input data 1 mahasiswa, tampilan prediksi IPK, grafik bar pengaruh faktor, dan penutup.", table_body_style),
        ],
    ]

    t = Table(table_data, colWidths=[4.2 * cm, 12.8 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Berkas PDF revisi berhasil dibuat: {output_filename}")


if __name__ == "__main__":
    create_proposal_pdf()
