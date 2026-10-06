"""Convert display equations stored as raw LaTeX text into editable Word OMML.

The report was authored with equation strings in ordinary paragraphs.  This
script parses the small LaTeX subset used by the report, emits MathML, and uses
Microsoft Office's MML2OMML transform to create native Word equations.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "reports" / "midterm" / "Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi.docx"
OUTPUT = SOURCE.with_name("Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi_cong_thuc_da_chinh.docx")
MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")

MATHML = "http://www.w3.org/1998/Math/MathML"


def m(tag: str, text: str | None = None, **attrs):
    node = etree.Element(f"{{{MATHML}}}{tag}", **attrs)
    if text is not None:
        node.text = text
    return node


COMMANDS = {
    "sum": ("mo", "∑"),
    "log": ("mi", "log"),
    "cos": ("mi", "cos"),
    "sin": ("mi", "sin"),
    "exp": ("mi", "exp"),
    "arg": ("mi", "arg"),
    "min": ("mi", "min"),
    "max": ("mi", "max"),
    "sign": ("mi", "sign"),
    "RMS": ("mi", "RMS"),
    "ZCR": ("mi", "ZCR"),
    "Mel": ("mi", "Mel"),
    "mu": ("mi", "μ"),
    "sigma": ("mi", "σ"),
    "pi": ("mi", "π"),
    "theta": ("mi", "θ"),
    "Delta": ("mi", "Δ"),
    "times": ("mo", "×"),
    "cdot": ("mo", "·"),
    "rightarrow": ("mo", "→"),
    "geq": ("mo", "≥"),
    "leq": ("mo", "≤"),
    "in": ("mo", "∈"),
    "mid": ("mo", "|"),
    "pm": ("mo", "±"),
    "ldots": ("mo", "…"),
    "lfloor": ("mo", "⌊"),
    "rfloor": ("mo", "⌋"),
    "Vert": ("mo", "‖"),
}


class LatexSubsetParser:
    def __init__(self, source: str):
        self.s = source.strip()
        self.i = 0

    def parse(self):
        return self.sequence()

    def sequence(self, stop: str | None = None):
        row = m("mrow")
        while self.i < len(self.s):
            if stop and self.s[self.i] == stop:
                self.i += 1
                break
            if self.s[self.i].isspace():
                self.i += 1
                continue
            atom = self.atom()
            if atom is None:
                continue
            sub = sup = None
            while self.i < len(self.s) and self.s[self.i] in "_^":
                kind = self.s[self.i]
                self.i += 1
                val = self.group_or_atom()
                if kind == "_":
                    sub = val
                else:
                    sup = val
            if sub is not None and sup is not None:
                wrap = m("msubsup")
                wrap.extend([atom, sub, sup])
                atom = wrap
            elif sub is not None:
                wrap = m("msub")
                wrap.extend([atom, sub])
                atom = wrap
            elif sup is not None:
                wrap = m("msup")
                wrap.extend([atom, sup])
                atom = wrap
            row.append(atom)
        return row

    def group_or_atom(self):
        if self.i < len(self.s) and self.s[self.i] == "{":
            self.i += 1
            return self.sequence("}")
        atom = self.atom()
        return atom if atom is not None else m("mrow")

    def raw_group(self) -> str:
        if self.i >= len(self.s) or self.s[self.i] != "{":
            return ""
        self.i += 1
        start = self.i
        depth = 1
        while self.i < len(self.s) and depth:
            if self.s[self.i] == "{":
                depth += 1
            elif self.s[self.i] == "}":
                depth -= 1
                if depth == 0:
                    value = self.s[start:self.i]
                    self.i += 1
                    return value
            self.i += 1
        return self.s[start:]

    def atom(self):
        if self.i >= len(self.s):
            return None
        ch = self.s[self.i]
        if ch == "{":
            self.i += 1
            return self.sequence("}")
        if ch == "\\":
            return self.command()
        if ch.isdigit() or ch in ".,":
            start = self.i
            while self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] in ".,"):
                self.i += 1
            return m("mn", self.s[start:self.i])
        if ch.isalpha():
            start = self.i
            while self.i < len(self.s) and self.s[self.i].isalpha():
                self.i += 1
            token = self.s[start:self.i]
            return m("mi", token, mathvariant="normal" if len(token) > 1 else "italic")
        self.i += 1
        if ch == "-":
            ch = "−"
        if ch == "*":
            ch = "×"
        return m("mo", ch)

    def command(self):
        self.i += 1
        if self.i < len(self.s) and not self.s[self.i].isalpha():
            ch = self.s[self.i]
            self.i += 1
            if ch == "|":
                # Treat \|...\| as one norm expression.  Separate bar tokens can
                # be collapsed incorrectly by Word's MathML-to-OMML transform.
                close = self.s.find(r"\|", self.i)
                if close != -1:
                    inner = LatexSubsetParser(self.s[self.i:close]).parse()
                    self.i = close + 2
                    fenced = m("mfenced", open="‖", close="‖")
                    fenced.append(inner)
                    return fenced
                return m("mo", "‖")
            return m("mo", ch)
        start = self.i
        while self.i < len(self.s) and self.s[self.i].isalpha():
            self.i += 1
        name = self.s[start:self.i]
        if name in ("left", "right"):
            return self.atom()
        if name in ("quad", "qquad"):
            return m("mspace", width="2em" if name == "qquad" else "1em")
        if name in (",", ";", "!"):
            return m("mspace", width="0.25em")
        if name == "frac":
            node = m("mfrac")
            node.extend([self.group_or_atom(), self.group_or_atom()])
            return node
        if name == "sqrt":
            node = m("msqrt")
            node.append(self.group_or_atom())
            return node
        if name == "text":
            return m("mtext", self.raw_group().replace("~", " "))
        if name in ("mathbb", "mathcal", "mathrm"):
            variant = {"mathbb": "double-struck", "mathcal": "script", "mathrm": "normal"}[name]
            node = m("mstyle", mathvariant=variant)
            node.append(self.group_or_atom())
            return node
        if name == "hat":
            node = m("mover", accent="true")
            node.extend([self.group_or_atom(), m("mo", "ˆ")])
            return node
        if name in COMMANDS:
            tag, value = COMMANDS[name]
            return m(tag, value, mathvariant="normal" if tag == "mi" and len(value) > 1 else "italic")
        # Unknown commands remain readable instead of disappearing.
        return m("mi", name, mathvariant="normal")


def normalize_formula(text: str) -> str:
    corrections = {
        r"T = \left\lfloor \frac{16000 - 640}{320} \right\rfloor + 1 = 48 + 1 = 49 \rightarrow 51\text{ frames (sau padding)}":
            r"T = \left\lfloor \frac{16000}{320} \right\rfloor + 1 = 51\text{ frames}\quad(\text{center}=\text{True})",
        r"ZCR = \frac{1}{2N} \sum_{n=1}^{N-1} |\text{sign}(x[n]) - \text{sign}(x[n-1])|":
            r"ZCR = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sign}(x[n]) - \text{sign}(x[n-1])|",
    }
    return corrections.get(text.strip(), text.strip())


def is_display_equation(paragraph) -> bool:
    text = paragraph.text.strip()
    if not text or paragraph.alignment != WD_ALIGN_PARAGRAPH.CENTER:
        return False
    # Captions and the ASCII flow diagram are centered too, but are not math.
    if text.startswith("Hình ") or "┌" in text or "│" in text:
        return False
    return bool(re.search(r"\\[A-Za-z]+|[_^]", text))


def mathml_to_omml(latex: str, transform):
    math = m("math")
    math.append(LatexSubsetParser(latex).parse())
    result = transform(math)
    root = result.getroot()
    return deepcopy(root)


def replace_paragraph_with_equation(paragraph, omath):
    p = paragraph._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    omath_para = OxmlElement("m:oMathPara")
    para_props = OxmlElement("m:oMathParaPr")
    justification = OxmlElement("m:jc")
    justification.set(qn("m:val"), "center")
    para_props.append(justification)
    omath_para.extend([para_props, omath])
    p.append(omath_para)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not MML2OMML.exists():
        raise FileNotFoundError(MML2OMML)

    doc = Document(SOURCE)
    transform = etree.XSLT(etree.parse(str(MML2OMML)))
    converted = []
    for index, paragraph in enumerate(doc.paragraphs):
        if not is_display_equation(paragraph):
            continue
        original = paragraph.text.strip()
        formula = normalize_formula(original)
        omath = mathml_to_omml(formula, transform)
        replace_paragraph_with_equation(paragraph, omath)
        converted.append((index, original, formula))

    doc.save(OUTPUT)
    print(f"Saved: {OUTPUT}")
    print(f"Converted display equations: {len(converted)}")
    for index, original, formula in converted:
        note = " [corrected]" if original != formula else ""
        print(f"  paragraph {index}: {original}{note}")


if __name__ == "__main__":
    main()
