from helpers import source
import pytest

from mimirag import extraction
from mimirag.datatypes import Page
from mimirag.errors import IntegrityError, MimiRAGError


def test_html_text_keeps_paragraphs_and_table_rows():
    html = (
        b'<html><head><style>p {color: red}</style><script>var x = 1;</script></head><body>'
        b'<h1>Title</h1><p>First   paragraph,\n  two lines.</p><table><tr><th>Key</th>'
        b'<td>Value &amp; more</td></tr><tr><td>A</td><td>B</td></tr></table></body></html>'
    )
    expected = 'Title\n\nFirst paragraph,\ntwo lines.\n\n| Key | Value & more\n\n| A | B'
    assert extraction.html_text(html) == expected


def test_pdf_pages_keep_blank_pages_and_select_one_page(make_pdf):
    path = make_pdf('doc.pdf', [['One'], [], ['Three']])
    pages = extraction.pdf_pages(path)
    assert len(pages) == 3
    assert pages[1].strip() == ''
    assert pages[2].startswith('Three')
    assert extraction.pdf_pages(path, 3) == [pages[2]]


def test_banners_are_repeated_margin_lines_and_only_margins_are_blanked():
    pages = [f'Manual\nbody {i}\nManual\nPage {i}' for i in range(1, 4)] + ['Manual\nlast body']
    banners = extraction.banner_lines(pages)
    assert banners == {'Manual'}
    assert extraction.blank_banners(pages[0], banners) == '\nbody 1\nManual\nPage 1'
    # Two pages are not enough, however large the share.
    assert extraction.banner_lines(['Header\nx', 'Header\ny']) == set()


def test_text_and_html_extraction_with_admitted_lines(tmp_path):
    (tmp_path / 'a.md').write_text('one\ntwo\nthree\nfour\n')
    (tmp_path / 'b.html').write_text('<p>Hello</p>')
    entry = source(tmp_path, 'a.md', admitted_lines=[[2, 2], [4, 4]])
    assert extraction.extract_document(tmp_path, entry) == ([Page(None, '\ntwo\n\nfour')], None)
    html = extraction.extract_document(tmp_path, source(tmp_path, 'b.html', kind='html'))
    assert html.pages == [Page(None, 'Hello')]
    assert html.banners is None
    with pytest.raises(MimiRAGError, match='outside source'):
        extraction.extract_document(tmp_path, source(tmp_path, 'a.md', admitted_lines=[[3, 9]]))


def test_pdf_extraction_removes_banners_and_merges_ocr_pages(tmp_path, make_pdf, ocr):
    pages = [
        ['Guide', 'Alpha body', '', 'p1'],
        [],
        ['Guide', 'Gamma body', '', 'p3'],
        ['Guide', 'Delta body', '', 'p4'],
    ]
    make_pdf('doc.pdf', pages)
    entry = source(tmp_path, 'doc.pdf', kind='pdf', pages=4, ocr_pages=[2])
    result = extraction.extract_document(tmp_path, entry, ocr)
    assert result.banners == ['Guide']
    assert [page.number for page in result.pages] == [1, 2, 3, 4]
    assert result.pages[0].text.startswith('\nAlpha body\n')
    assert result.pages[1].text == 'Recognized text of p2\nsecond line'
    with pytest.raises(MimiRAGError, match='no OCR command'):
        extraction.extract_document(tmp_path, entry)
    for bad in ([5], [2, 2], ['2']):
        with pytest.raises(MimiRAGError, match='unique PDF page numbers'):
            extraction.extract_document(tmp_path, dict(entry, ocr_pages=bad), ocr)
    with pytest.raises(IntegrityError, match='page-count'):
        extraction.extract_document(tmp_path, dict(entry, pages=3), ocr)


def test_extract_page_matches_the_whole_document_output(tmp_path, make_pdf, ocr):
    make_pdf('doc.pdf', [['Guide', 'Alpha'], [], ['Guide', 'Gamma'], ['Guide', 'Delta']])
    entry = source(tmp_path, 'doc.pdf', kind='pdf', ocr_pages=[2])
    pages, banners = extraction.extract_document(tmp_path, entry, ocr)
    for number, text in pages:
        assert extraction.extract_page(tmp_path, entry, number, banners, ocr) == text
    with pytest.raises(MimiRAGError, match='start at 1'):
        extraction.extract_page(tmp_path, entry, 0, banners, ocr)
    with pytest.raises(MimiRAGError, match='no OCR command'):
        extraction.extract_page(tmp_path, entry, 2, banners)


def test_ocr_command_sees_its_image_directory_and_must_return_one_text_per_page(tmp_path, make_pdf, ocr):
    path = make_pdf('doc.pdf', [[], ['b'], []])
    seen = tmp_path / 'seen.txt'
    script = tmp_path / 'capture.py'
    script.write_text(
        'import sys, pathlib\n'
        f'pathlib.Path({str(seen)!r}).write_text(pathlib.Path(sys.argv[1]).read_text())\n'
        "sys.stdout.write('x\\fy')\n"
    )
    command = [ocr['command'][0], str(script), '{images}/list.txt']
    capture = dict(ocr, command=command, images_at='/in')
    assert extraction.ocr_pages(path, [1, 3], capture) == {1: 'x', 3: 'y'}
    assert seen.read_text() == '/in/p1.png\n/in/p3.png\n'
    with pytest.raises(MimiRAGError, match='one text per page'):
        extraction.ocr_pages(path, [1], capture)


def test_undecodable_text_and_unreadable_pdfs_are_refusals(tmp_path):
    (tmp_path / 'latin.txt').write_bytes('caf\xe9'.encode('latin-1'))
    with pytest.raises(MimiRAGError, match='not UTF-8'):
        extraction.extract_document(tmp_path, source(tmp_path, 'latin.txt'))
    (tmp_path / 'broken.pdf').write_bytes(b'%PDF-1.4 not really a PDF')
    with pytest.raises(MimiRAGError, match='pdftotext failed'):
        extraction.extract_document(tmp_path, source(tmp_path, 'broken.pdf', kind='pdf'))
